# mistral.rs GPU/CPU 分层推理源码更改报告

## 概述

mistral.rs 在 v0.8.23 → v0.9.0 版本迭代中，新增了 **异构设备分层推理（GPU/CPU Split Inference）** 功能。该功能允许将 Transformer 模型的解码器层（decoder layers）分布到不同的计算设备上——GPU、CPU、乃至**远端机器（通过 TCP）**——实现跨设备的协同推理。

本报告基于对 `mistral.rs` 源码仓库的全面分析（特别是 13 个关键 commit，从 `e527fbeb7` 到 `8f41e2e1e`），详细记录了为支持此功能所做的一切源码变更。

---

## 核心架构：三大分层策略

| 策略 | 机制 | 适用场景 |
|------|------|---------|
| **Layer Mapping (P2P)** | 将连续的层范围放置在不同设备上，激活值仅在层边界跨设备传输 | 单机多 GPU、GPU+CPU 混合 |
| **Tensor Parallelism (NCCL/Ring)** | 每层切分到所有 GPU，每设备持有每层的一部分参数 | 单机/多机 NCCL 全并行 |
| **Remote Layer Offloading (TCP)** | 将连续的层块委托给远端 TCP worker 进程计算 | 手机↔PC、多机分布式 |

---

## 一、新增模块

### 1.1 `mistralrs-core/src/device_map/remote.rs` — 远端层推理（全新文件，332 行）

这是 GPU/CPU 分层推理最核心的新模块，实现了**跨机器 TCP 层卸载**。

#### `RemoteConnectionPool` (行 17-138)
- 管理到远端 worker 的持久化 TCP 连接池（按地址字符串索引）
- 连接配置：`TCP_NODELAY`、300s 读超时、30s 写超时
- **线协议** (行 55-138)：
  ```
  请求: [1B cmd][4B LE layer_start][4B LE layer_end][4B LE past_kv][4B reserved][8B LE len][payload]
  响应: [8B LE len][response_bytes]
  ```
- 支持断线自动重连（行 119-137）

#### `serialize_tensor()` (行 142-158)
- 将张量序列化为字节流：`[8B num_elements][8B ndims][dims...][f32 data]`
- 强制转为 CPU F32 以确保跨设备兼容

#### `deserialize_tensor()` (行 161-186)
- 从字节流反序列化为目标设备上的张量

#### `RemoteLayerMapper` (行 190-370)
- 实现 `DeviceMapper` trait 的**复合映射器**
- **核心结构**（行 190-206）：
  - `local_mapper: LayerDeviceMapper` — 处理本地层（GPU/CPU）
  - `connection_pool: RemoteConnectionPool` — 处理远端层（TCP）
  - `remote_blocks: Vec<(addr, first_layer, last_layer)>` — 预计算的连续远端块
  - `layer_to_block: Vec<Option<usize>>` — 每层到远端块的索引映射
  - `last_remote_block: AtomicUsize` — 跟踪上一个计算的远端块（避免重复计算）
  - `past_kv: AtomicU32` — KV 缓存位置（用于 RoPE 正确计算）
- **连续远端块优化**（行 216-238）：启动时扫描所有层，将连续的同地址远端层合并为一个块
- **`map()` 方法**（行 266-297）— 关键性能优化：
  - 仅在远程块的**第一层**触发 TCP 往返
  - 同一块内的后续层直接透传（forward loop 已持有块输出）
  - 结果从 CPU 移回输入设备以保持设备一致性
- **`is_layer_remote()`**（行 361-365）：标记远端层，调用方据此跳过本地权重加载
- **`set_past_kv()`**（行 367-369）：传播 KV 位置到远端 worker，确保 RoPE 位置编码正确

### 1.2 `mistralrs-cli/src/remote_worker.rs` — TCP 远端 Worker（全新文件，195 行）

远端机器上运行的独立二进制程序，加载完整模型（CPU），监听 TCP，按需计算指定层范围。

- **CMD_COMPUTE (0x00)**：接收隐藏状态 → 调用 `pipeline.forward_from_layer(hidden, start, end, past_kv)` → 返回计算结果
- **CMD_RESET (0x01)**：调用 `pipeline.reset_kv_cache()` 重置 KV 缓存
- 每个 TCP 连接在独立线程中处理（支持多客户端并发）
- 模型通过 `Arc<tokio::sync::Mutex<dyn Pipeline>>` 共享，使用 `try_lock()` 避免阻塞

---

## 二、核心接口变更

### 2.1 `DeviceMapper` trait 新增方法

**文件**: `mistralrs-core/src/device_map/mappers.rs` (行 9-36)

```rust
// 新增方法:
fn is_layer_remote(&self, _layer: usize) -> bool { false }  // 默认: 不是远端
fn set_past_kv(&self, _past_kv: u32) {}                     // 默认: 空操作
```

- **`is_layer_remote()`**：判断某层的权重是否在远端机器上，调用方据此跳过本地权重加载
- **`set_past_kv()`**：在推理前设置当前 KV 缓存位置，远端映射器需要此信息来正确计算 RoPE

### 2.2 `Pipeline` trait 新增方法

**文件**: `mistralrs-core/src/pipeline/mod.rs` (行 1836-1849)

```rust
fn forward_from_layer(&self, _hidden: &Tensor, _start_layer: usize,
    _end_layer: usize, _past_kv_len: usize) -> Result<Tensor> {
    bail!("forward_from_layer not implemented for this pipeline")
}

fn reset_kv_cache(&self) -> Result<()> {
    bail!("reset_kv_cache not implemented for this pipeline")
}
```

- **`forward_from_layer()`**：从预计算的隐藏状态开始，仅执行指定层范围的 forward pass，供远端 worker 使用
- **`reset_kv_cache()`**：重置 KV 缓存，供远端 worker 在收到 CMD_RESET 时调用

### 2.3 `Topology` 新增 `RemoteAwareDevice` 枚举

**文件**: `mistralrs-core/src/topology/mod.rs` (行 24-42)

```rust
pub enum RemoteAwareDevice {
    Local(Device),           // 本地设备 (Cpu/Cuda/Metal)
    Remote { addr: String }, // 远端 TCP worker (例: "192.168.1.100:5050")
}
```

- 新增 `remote:tcp://IP:PORT` 设备格式支持（行 13, 156-160）
- 新增 `has_remote_devices()` 方法（行 113-123）检测拓扑中是否包含远端设备
- 新增 `immediate_overrides()` 将拓扑编译为 ISQ 量化覆盖列表（行 282-308）

### 2.4 `DeviceMapMetadata` 新增 `topology_mapper()` 方法

**文件**: `mistralrs-core/src/device_map/mod.rs` (行 129-216)

- 当存在远端设备时，构建 `RemoteLayerMapper`
- 当全部本地时，回退到标准的 `LayerDeviceMapper`
- 本地设备使用当前设备的 `DeviceId` 进行规范化（行 167-175），确保 CUDA kernel 中的 `same_device()` 比较不失败（修复 RoPE 不匹配问题）

---

## 三、模型层变更（7 个 GGUF 量化模型文件）

所有 7 个量化模型文件都进行了三处一致的修改：

### 3.1 权重加载时跳过远端层

```rust
// 示例: quantized_llama.rs 行 479-486
if mapper.is_layer_remote(layer_idx) {
    layers.push(None);     // 远端层: 不加载本地权重
    continue;
}
let Some(device) = mapper.device_for(layer_idx, false) else {
    layers.push(None);     // device_for 返回 None 意味着权重不在本地
    continue;
};
```

### 3.2 推理前设置 past_kv

```rust
// 示例: quantized_llama.rs 行 704-706
if let Some(ref mapper) = self.mapper {
    let past_kv = start_offsets.first().copied().unwrap_or(0) as u32;
    mapper.set_past_kv(past_kv);
}
```

### 3.3 推理时每层 map 激活值到对应设备

```rust
// 示例: quantized_llama.rs 行 708-710
for (i, layer) in self.layers.iter().enumerate() {
    if let Some(ref mapper) = self.mapper {
        layer_in = mapper.map(layer_in, i)?;  // 移动到该层的设备
    }
    // ... layer forward ...
}
```

### 3.4 新增 `forward_from_layer()` 方法（4 个模型）

以下模型实现了供远端 worker 使用的 `forward_from_layer()`：

| 模型文件 | `forward_from_layer` 行号 |
|---------|--------------------------|
| `quantized_llama.rs` | 746-796 |
| `quantized_qwen.rs` | 517-567 |
| `quantized_qwen3.rs` | ~444-494 |
| `quantized_qwen3_moe.rs` | ~1104-1154 |

这些方法：
- 从指定的 `start_layer` 开始执行 forward pass
- 到 `end_layer`（含）结束
- 跳过来自 `None` 层的层（远端 worker 持有所有权重，所以无 None 层）
- 对 `seq_len > 1` 的 prompt tokens 使用因果注意力掩码（`d3e7b498c` 修复）
- 将输出移回 CPU 以便 TCP 传输

### 3.5 修改的文件完整列表

| 文件 | 行数变更 |
|------|---------|
| `models/quantized_llama.rs` | +60 / -9 (skip remote + forward_from_layer) |
| `models/quantized_qwen.rs` | +55 / -7 |
| `models/quantized_qwen3.rs` | +55 / -7 |
| `models/quantized_qwen3_moe.rs` | +55 / -7 |
| `models/quantized_phi2.rs` | +14 / -3 (skip remote only) |
| `models/quantized_phi3.rs` | +14 / -3 (skip remote only) |
| `models/quantized_starcoder2.rs` | +18 / -3 (skip remote only) |

---

## 四、Pipeline 层变更

### 4.1 GGUF Pipeline — 远端 worker 支持

**文件**: `mistralrs-core/src/pipeline/gguf.rs`

- **`GGUFPipeline::forward_from_layer()`**（行 88-103）：按模型架构分发到对应的 `Model::forward_from_layer()`。支持 Llama、Qwen2、Qwen3、Qwen3MoE 四种架构
- **`GGUFPipeline` 的 `Pipeline` trait 实现**（行 856-870）：
  - `forward_from_layer()`：获取 KV 缓存锁 → 调用模型层方法
  - `reset_kv_cache()`：清除所有层的 KV 缓存
- **GGUF 加载器自动设备映射**（行 347-369）：当 `DeviceMapSetting::Auto` 时使用 `GgufDeviceMapLoaderInner` 适配器计算自动层分布

### 4.2 Normal/Multimodal/Embedding Pipeline — 自动设备映射集成

**文件**: `mistralrs-core/src/pipeline/normal.rs` (行 343+)，`multimodal.rs` (行 263+)，`embedding.rs` (行 180+)

加载流程统一为：
1. 枚举可用设备 (`get_all_similar_devices()`)
2. 若 `DeviceMapSetting::Auto` → 调用 `auto_device_map::get_device_layers()` 计算最优分布
3. 构建 `DeviceMapper` 实现
4. 记录每层的设备
5. 如果有 CPU 层则禁用 paged attention
6. 将 mapper 存入 pipeline 结构

### 4.3 `DeviceMappedModelLoader` trait

**文件**: `mistralrs-core/src/pipeline/loaders/mod.rs` (行 561-627)

定义了自动设备映射所需的方法：
- `layer_sizes_in_bytes()` — 每个解码器层的字节大小
- `non_mapped_size_in_bytes()` — 非映射部分（embedding、lm_head）的大小
- `num_layers()` — 解码器层总数
- `get_device_layers()` — 默认委托给 `auto_device_map::get_device_layers()`

### 4.4 自动设备映射算法

**文件**: `mistralrs-core/src/pipeline/loaders/auto_device_map.rs` (行 192-426)

- 查询每个设备可用内存（`MemoryUsage`）
- 2% VRAM 安全余量 + 512 MB 最小缓冲（行 15-16）
- 预留 KV 缓存和激活值所需内存
- 优先在可用内存最多的设备上放置层
- 若某设备能容纳所有剩余层则一次性全部放置
- 否则尽可能多地打包层
- GPU 优先，CPU 作为回退

---

## 五、配套基础设施

### 5.1 `DeviceMappedMask`

**文件**: `mistralrs-core/src/device_map/mask.rs`

- 为每个唯一设备预先创建一个注意力掩码副本
- 推理时通过 `mask.get(device)` 获取，避免重复分配和跨设备传输

### 5.2 `CudaPeerAccess`

**文件**: `mistralrs-core/src/device_map/peer.rs`

- 查询 CUDA 驱动以获取 GPU 对之间的 P2P 访问能力
- 有 P2P 时直接 GPU-to-GPU 传输，无 P2P 时通过 CPU 中转

### 5.3 KV 缓存 `Option<Device>` 支持

**文件**: `mistralrs-core/src/kv_cache/mod.rs`（commit `6963bce9f`）

- KV 缓存增加 `Option<Device>` 支持，使远端层的缓存条目可正确地在目标设备上创建

### 5.4 CLI 入口

**文件**: `mistralrs-cli/src/args/mod.rs` (行 238-256)

新增 `remote-worker` 子命令：
```
mistralrs remote-worker --model-dir . --model-file model.gguf --listen 0.0.0.0:5050 --layers "10-17"
```

**文件**: `mistralrs-cli/src/main.rs` — 添加 `Commands::RemoteWorker` 分支调度到 `remote_worker::run_remote_worker()`

---

## 六、关键 Bug 修复轨迹

| Commit | 问题 | 修复 |
|--------|------|------|
| `8be5f1cc4` | Qwen 模型 RoPE 预创建时重复 `layers.push(None)` | 删除重复的 push |
| `51b79e35f` | 单线程 worker 性能瓶颈 | 多线程 worker + 单一 `RemoteConnectionPool` |
| `6963bce9f` | KV 缓存 `Option<Device>` 导致远端层出错 | 扩展 KV 缓存以支持 `Option<Device>` |
| `f73e6970a` | 每个远端层都触发一次 TCP 往返 | 合并为每个远端块一次往返 + 修复拓扑范围解析 |
| `03cf2b065` | past_kv 位置未传到远端 worker，RoPE 错误 | 通过 `DeviceMapper::set_past_kv()` 传播位置 |
| `182fb57d6` | 远端输出张量留在 CPU 上导致后续层设备不匹配 | `map()` 中将结果移回输入设备 |
| `d3e7b498c` | `forward_from_layer` 对 prompt tokens 未使用因果掩码 | 对 seq_len > 1 应用因果掩码 |
| `8f41e2e1e` | CUDA `DeviceId` 差异导致 `same_device()` 比较失败 | 在拓扑构建时规范化 CUDA `DeviceId` |

---

## 七、配置与使用示例

### 7.1 拓扑 YAML 格式

**文件**: `topologies/isq_and_device.yml`

```yaml
0-8:
  isq: Q3K
  device: cuda[0]
8-16:
  isq: Q4K
  device: cpu
16-24:
  isq: Q6K
  device: remote:tcp://192.168.1.100:5050
28-32:
  isq: Q8_0
  device: cuda[0]
```

支持：
- 范围选择器：`0-16: device: cuda[0]`
- 单层选择：`12: isq: Q4K`
- 正则匹配：`/model\.layers\.\d+\.self_attn\..*/`
- 远端设备：`device: remote:tcp://IP:PORT`

### 7.2 CLI 手动映射

```bash
mistralrs run -m model.gguf --device-layers "0:16;1:8" # GPU0 上 16 层, GPU1 上 8 层, 剩余在 CPU
```

### 7.3 Rust SDK 自动映射

```rust
let mapper = DeviceMapSetting::Auto(AutoDeviceMapParams::Text {
    max_seq_len: 4096,
    max_batch_size: 1,
});
```

---

## 八、完整变更文件清单

### 新增文件（3 个）
| 文件 | 行数 | 用途 |
|------|------|------|
| `mistralrs-core/src/device_map/remote.rs` | 371 | RemoteLayerMapper, RemoteConnectionPool, 序列化/反序列化 |
| `mistralrs-cli/src/remote_worker.rs` | 195 | TCP 远端 worker 守护进程 |
| `topologies/isq_and_device.yml` | 12 | 拓扑配置示例 |

### 修改文件（25 个）
| 文件 | 主要变更 |
|------|---------|
| `mistralrs-core/src/device_map/mod.rs` | topology_mapper(), DeviceId 规范化, 远端设备支持 |
| `mistralrs-core/src/device_map/mappers.rs` | DeviceMapper trait 新增 is_layer_remote(), set_past_kv() |
| `mistralrs-core/src/topology/mod.rs` | RemoteAwareDevice 枚举, remote:tcp:// 解析 |
| `mistralrs-core/src/pipeline/mod.rs` | Pipeline trait 新增 forward_from_layer(), reset_kv_cache() |
| `mistralrs-core/src/pipeline/gguf.rs` | GGUFPipeline::forward_from_layer(), 自动映射集成 |
| `mistralrs-core/src/pipeline/normal.rs` | 加载流程集成设备映射器 |
| `mistralrs-core/src/pipeline/multimodal.rs` | 同 normal.rs |
| `mistralrs-core/src/pipeline/embedding.rs` | 同 normal.rs |
| `mistralrs-core/src/pipeline/loaders/mod.rs` | DeviceMappedModelLoader trait |
| `mistralrs-core/src/pipeline/loaders/auto_device_map.rs` | 自动层放置算法 |
| `mistralrs-core/src/pipeline/loaders/normal_loaders.rs` | ~30+ 模型 loader 实现 DeviceMappedModelLoader |
| `mistralrs-core/src/pipeline/loaders/multimodal_loaders.rs` | 多模态 loader |
| `mistralrs-core/src/pipeline/loaders/embedding_loaders.rs` | Embedding loader |
| `mistralrs-core/src/models/quantized_llama.rs` | 跳过远端层 + forward_from_layer |
| `mistralrs-core/src/models/quantized_qwen.rs` | 跳过远端层 + forward_from_layer |
| `mistralrs-core/src/models/quantized_qwen3.rs` | 跳过远端层 + forward_from_layer |
| `mistralrs-core/src/models/quantized_qwen3_moe.rs` | 跳过远端层 + forward_from_layer |
| `mistralrs-core/src/models/quantized_phi2.rs` | 跳过远端层 |
| `mistralrs-core/src/models/quantized_phi3.rs` | 跳过远端层 |
| `mistralrs-core/src/models/quantized_starcoder2.rs` | 跳过远端层 |
| `mistralrs-core/src/kv_cache/mod.rs` | Option<Device> 支持 |
| `mistralrs-core/src/tuning.rs` | 自动调优 Hybrid fit 检测 |
| `mistralrs-cli/src/args/mod.rs` | remote-worker 子命令 |
| `mistralrs-cli/src/main.rs` | remote-worker 分支调度 |

### 新增示例 / 文档
| 文件 | 用途 |
|------|------|
| `examples/advanced/auto_device_map/main.rs` | Rust SDK 自动设备映射示例 |
| `examples/quantization/topology/main.rs` | Rust 拓扑使用示例 |
| `examples/python/text_auto_device_map.py` | Python 自动映射示例 |
| `examples/python/multimodal_auto_device_map.py` | Python 多模态映射示例 |
| `examples/python/topology.py` | Python 拓扑示例 |
| `docs/src/content/docs/guides/perf/distributed-inference.mdx` | 分布式推理指南 |
| `docs/src/content/docs/guides/perf/topology.mdx` | 拓扑配置指南 |

---

## 九、数据流：端到端 GPU/CPU 分层推理

```
用户配置 (--device-layers / Topology / AutoDeviceMap)
    │
    ▼
加载阶段 (Pipeline::load_model_from_path)
    │
    ├─ get_all_similar_devices() → 枚举可用 GPU
    ├─ DeviceMapSetting::Auto? → auto_device_map::get_device_layers()
    │     └─ 查询每设备内存 → 迭代打包层 → 计算 DeviceMapMetadata
    ├─ Topology 存在? → topology_mapper()
    │     ├─ 有 remote? → RemoteLayerMapper
    │     └─ 全本地?  → LayerDeviceMapper
    ├─ 无上述? → manual_mappings() + LayerDeviceMapper
    └─ mapper 存入 pipeline
    │
    ▼
权重加载 (Model::new / Model::load)
    │
    ├─ mapper.is_layer_remote(i) → layers.push(None), 跳过加载
    ├─ mapper.device_for(i) → 获取目标设备
    └─ mapper.set_device(i, vb) → 指导 VarBuilder 加载到正确设备
    │
    ▼
推理 (Model::forward)
    │
    ├─ mapper.set_past_kv(past_kv) → 设置 KV 位置
    ├─ for each layer:
    │     ├─ mapper.map(x, i) → 移动激活值到该层设备
    │     │     ├─ 本地: CudaPeerAccess::to_device() (P2P 或 CPU 中转)
    │     │     └─ 远端: RemoteLayerMapper::map()
    │     │           ├─ 块首层: 序列化 → TCP roundtrip → 反序列化
    │     │           └─ 块内: 直接透传 (forward loop 已有结果)
    │     ├─ layer.forward() → 执行注意力 + MLP
    │     └─ 下一层迭代
    └─ 输出
```

---

## 十、架构设计要点总结

1. **层次化 DeviceMapper trait**：`DummyDeviceMapper`（单设备）→ `LayerDeviceMapper`（多本地设备 + P2P）→ `RemoteLayerMapper`（混合本地/远端）→ `NcclDeviceMapper`（张量并行）。每层增加能力而不破坏已有代码。

2. **TCP 往返最小化**：通过预计算连续远端块，仅在块边界触发 TCP 通信，而非每层一次往返。这是通过 `last_remote_block` 原子变量和 forward loop 语义实现的。

3. **权重加载与推理分离**：远端层的权重不加载到本地（`layers.push(None)`），但推理时激活值正常流经（由 `map()` 处理传输）。

4. **设备一致性保障**：通过 DeviceId 规范化、输出张量回移、因果掩码显式设置，确保跨设备张量操作的正确性。

5. **向后兼容**：所有新方法在 trait 中都有默认实现（`is_layer_remote() → false`，`set_past_kv() → noop`，`forward_from_layer() → bail!`），未使用该功能的模型完全不受影响。
