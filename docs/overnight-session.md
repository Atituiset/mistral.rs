# 跨机异构推理 — 全历史总结报告

**覆盖时间**: 2026-07-07 ~ 2026-07-25 (跨越 18 天, 9+ 个 Claude Code session)
**目标**: GPU PC (RTX 4050 6GB) + WSL PC (CPU 15GB) + Mate 40 Pro 手机三方异构推理
**当前阶段**: mistral.rs TCP 桥接分层推理, 部署 Qwen3.6-35B-A3B (16GB GGUF)
**总 Commits**: 16 (mistral.rs bridge + SSM), 另有 llama.cpp 3-machine 分支 commits

---

## 一、全项目时间线 (跨 Session)

### 第一阶段: llama.cpp RPC 3-machine (7/7 ~ 7/15)

| 日期 | Session | 事件 |
|------|---------|------|
| 7/7 22:44 | `5fdae812` | 讨论异构推理可行性 (kv-cache 跨设备复用), 探索项目结构 |
| 7/8 06:46 | `44448800` | Cron 定时任务: 检查手机端 llama.cpp 编译状态, task_id 已过期 |
| 7/8 17:59 | `ea3d3ea0` | **上一 session 上下文窗口溢出**, 新 session 继续。讨论 PC+手机分层调度 DEBUG 日志 |
| 7/9 22:40 | `3b612191` | 尝试 3-machine SSH 组网: GPU PC + WSL + 手机 — **36 次 SSH 失败**, 手机不可达 |
| 7/10-14 | `a7d692f0` | **3 天 mega-session**: 无 GPU 环境下测试 Vulkan (MNN + ncnn_llm 基准), 142 错误, 15 次上下文压力 |
| 7/14 23:30 | `acfa560e` | GPU PC 恢复可用, 讨论 Qwen3.6-35B-A3B-FP8 部署策略, 选策略 A |
| 7/15 21:56 | `d8f2e78b` | 核对文档与日志一致性, 设计断点续传 + watchdog 通宵策略 |

**核心问题**: llama.cpp RPC 方式依赖 SSH 隧道, 手机频繁不可达, 上下文窗口反复溢出。

### 第二阶段: mistral.rs TCP 桥接 (7/24 21:56 ~ 7/25 06:07, 8 小时通宵)

| 时间 (CST) | 事件 |
|------------|------|
| 21:56 | Phase 1-2-4: TCP 设备语法 + RemoteLayerMapper |
| 22:05 | Phase 3: 7 个模型的 layers→Option 适配 |
| 22:09 | Phase 5-6: Remote worker 骨架 |
| 22:17 | forward_from_layer + remote worker 完整实现 |
| 00:28 | KV cache Option<Device> 支持 |
| 01:16 | 真实 remote worker: Qwen2/3/MoE forward_from_layer |
| 01:43 | 多线程 worker + RemoteConnectionPool |
| 01:59 | 修复 RoPE 重复 push(None) |
| 02:39 | 单次 roundtrip 优化 + topology 范围修复 |
| 02:57 | past_kv 传播 |
| 03:37 | Remote 输出设备搬移 |
| 04:04 | Causal mask 修复 |
| 04:44 | DeviceId 规范化修复 RoPE 不匹配 |
| 05:15 | Qwen35MoE GGUF 架构识别 |
| 05:49 | SSM (Gated DeltaNet) 完整实现 |
| 06:07 | ffn_norm fallback + PartialEq + 最终编译通过 |

---

## 二、Session 崩溃与中断记录

### 本 session (7/24-25) — `90098635`

#### 7 次 Compaction 事件 (全部手动/定时触发)

| # | 时间 (CST) | 触发方式 | 压缩前 Tokens | 压缩后 | 耗时 |
|---|-----------|---------|-------------|--------|------|
| 1 | 14:50 | 手动 `/compact` | 300,897 | 12,584 | 68s |
| 2 | 16:32 | 手动 | 174,671 | 16,075 | 70s |
| 3 | 18:41 | 自动 cron? | 164,597 | 15,964 | 61s |
| 4 | 19:39 | 自动 | 157,556 | 20,013 | 46s |
| 5 | 21:25 | 自动 | 208,918 | 11,334 | 57s |
| 6 | 23:00 | 手动 | 171,422 | 12,403 | 59s |

**总计清除约 118 万 tokens**。每 90 分钟触发一次。5 号压缩前曾达 20.9 万 tokens 峰值。

#### 21 次 Auto-Mode 分类器拦截

分类器模型 `deepseek-v4-pro` 不可用时, Bash/Agent 工具调用被拦截:

| 时间范围 (CST) | 次数 | 拦截的命令类型 |
|---------------|------|--------------|
| 20:54-20:55 | 2 | SSH 到 GPU 解压 CUDA 二进制 |
| 22:24-22:43 | 2 | tar 源码 + SSH cargo build |
| 23:52-23:59 | 2 | 安装 GPU 二进制 + hybrid topology 测试 |
| 00:44 | 1 | **Agent 调用被拦截** (桥接端到端测试) |
| 01:07 | 1 | 启动 remote worker |
| 02:45 | 1 | 启动 WSL remote worker |
| 03:19-03:21 | 2 | rsync 到 GPU |
| 04:59 | 1 | HuggingFace 下载 Qwen3.6-35B-A3B |
| 05:19-05:21 | 2 | rsync 源码到 GPU |
| 05:51 | 1 | 测试 Qwen3.6 模型加载 |

#### 3 次 Tool 不存在错误
- 20:25, 21:34 — `Glob` 工具不可用 (2 次)
- 03:33 — `Grep` 工具不可用

#### 11 次用户中断

| 时间 (CST) | 中断内容 |
|-----------|---------|
| 21:11 | Gemma4 下载检查 |
| 21:32 | Playwright 测试 |
| 22:48 | 本地 topology 测试, cron 安装 |
| 00:31 | WSL cargo build (连按两次) |
| 06:03 | WSL cargo build (连按两次) |
| 06:59-07:01 | Summary 写作中断 + Qwen3.6 测试中断 |

### GPU PC OOM 崩溃链 (7/25 06:03-06:45 CST)

这是整个通宵最严重的事故:

```
06:03 -- 用户中断 WSL cargo build, 报告 "GPU机器报了内存OOM了"
06:04 -- GPU PC 被迫重启, SSH "Connection closed by 192.168.1.10 port 22"
06:07 -- 强制 /compact (此时 171,422 tokens)
06:45 -- WSL 自身崩溃: "灾难性故障, 错误代码: Wsl/Service"
06:53 -- 用户重新连接, 要求总结
```

**根因分析**: GPU PC 只有 15GB RAM + 4GB swap, `cargo build --release --features cuda` 并行编译中 Rust 编译器消耗大量内存。同时可能有模型加载测试占用内存。内存耗尽触发 kernel OOM killer, 系统被迫重启。

### 跨 Session 崩溃记录 (7/7-7/15)

| Session | 崩溃类型 | 详情 |
|---------|---------|------|
| `ea3d3ea0` | 上下文窗口溢出 | 前一 session 超出限制, 用户明确说"上一个会话超上下文窗口了" |
| `44448800` | 后台任务过期 | `bhmeyi1fz` (手机端编译) 已经清理, 无法恢复状态 |
| `3b612191` | 36 次 SSH 失败 | GPU PC 密码认证拒绝, 手机完全不可达 |
| `a7d692f0` | 142 工具错误 / 15 次上下文压力 | 3 天 session 反复接近窗口限制 |
| `057fd8d3` | Session 截断 | Cron 定时任务, 2 个 tool call 未返回结果就被终止 |

---

## 三、所有 Commits (16 个, 按时间)

| # | Commit | 时间 (CST) | 描述 |
|---|--------|-----------|------|
| 1 | `e527fbeb7` | 7/24 21:56 | Phase 1-2-4: remote:tcp:// 拓扑解析 + DeviceMapper trait + RemoteConnectionPool |
| 2 | `9b9c5cb8d` | 7/24 22:05 | Phase 3: 7 个 GGUF 模型文件的 layers→Option<LayerWeights> 适配 |
| 3 | `0c60b16b7` | 7/24 22:09 | Phase 5-6 骨架: remote worker + Gemma4 GGUF 架构识别 |
| 4 | `a6fd248a6` | 7/24 22:17 | Phase 5-6: forward_from_layer 实现 + remote worker 完整逻辑 |
| 5 | `6963bce9f` | 7/25 00:28 | KV cache `Option<Device>` 支持 + 协议对齐 |
| 6 | `c1ae14fd3` | 7/25 01:16 | 真实 remote worker: Qwen2/3/MoE forward_from_layer |
| 7 | `51b79e35f` | 7/25 01:43 | 多线程 remote worker + 单一 RemoteConnectionPool |
| 8 | `8be5f1cc4` | 7/25 01:59 | 修复 RoPE 预创建循环中重复 push(None) |
| 9 | `f73e6970a` | 7/25 02:39 | 每个 remote block 单次 roundtrip + topology 范围修复 |
| 10 | `03cf2b065` | 7/25 02:57 | past_kv 通过 DeviceMapper 传播以产生正确 RoPE 位置 |
| 11 | `182fb57d6` | 7/25 03:37 | remote 输出 tensor 搬回输入设备 |
| 12 | `d3e7b498c` | 7/25 04:04 | forward_from_layer 中 prompt tokens 使用 causal mask |
| 13 | `8f41e2e1e` | 7/25 04:44 | DeviceId 规范化修复 RoPE CUDA 设备不匹配 |
| 14 | `ebb6734f2` | 7/25 05:15 | GGUF 架构添加 Qwen35MoE 识别 |
| 15 | `caa92ce20` | 7/25 05:49 | **SSM (Gated DeltaNet) 完整实现** (652→1246 行) |
| 16 | `f66ff28c2` | 7/25 06:07 | ffn_norm 回退 + PartialEq + SSM 感知 per_layer_size |

---

## 四、所有 Bug 与修复 (23 个)

### 桥接协议 Bug (13 个)

| # | 错误 | 症状 | 修复 | Commit |
|---|------|------|------|--------|
| 1 | Causal mask 缺失 | 桥接输出 "I don't understand" vs 本地 "2+2=4" | seq_len>1 时构建 causal mask | `d3e7b498c` |
| 2 | `[usize; 1]` trait bound | 编译错误: PastKvLenCache 未实现 | 中间变量 `let kv_ref: &[usize] = &kv_offsets` | `d3e7b498c` |
| 3 | `Tensor::zeros((1u32, seq_len))` | 类型不匹配: (u32, usize) 不是 Shape | 改用 `(1, seq_len)` | `d3e7b498c` |
| 4 | RoPE CUDA 设备不匹配 | "apply-rotary tensors must be on the same cuda device" | DeviceId 规范化 (两处) | `8f41e2e1e` |
| 5 | 重复 `layers.push(None)` | Layer 计数错误 | 从 RoPE 循环移除 push | `8be5f1cc4` |
| 6 | Remote 输出在错误设备 | 跨设备 tensor 操作失败 | `.to_device(&self.device)` | `182fb57d6` |
| 7 | past_kv 硬编码 0 | decode 阶段 RoPE 位置错误 | DeviceMapper trait 增加 `set_past_kv()` + AtomicU32 | `03cf2b065` |
| 8 | Topology 范围不匹配 | 层索引分配错误 | 修正 IoU 边界逻辑 | `f73e6970a` |
| 9 | 多线程连接冲突 | 多个 TCP 连接争用端口 | 单一 RemoteConnectionPool (HashMap<addr, TcpStream>) | `51b79e35f` |
| 10 | GGUF 架构 `qwen35moe` 未知 | 模型加载失败 | 添加 Qwen35MoE 到 enum | `ebb6734f2` |
| 11 | IQ2_M 量化不支持 | 模型加载失败 | 改用 Q3_K_M (17.1GB→16GB) | — |
| 12 | Gemma4 架构未识别 | remote worker 加载 Gemma4 失败 | 添加 Gemma4 到 GGUFArchitecture | `0c60b16b7` |
| 13 | KV cache Device 类型不匹配 | 编译错误 | KV cache 支持 Option<Device> | `6963bce9f` |

### SSM 实现 Bug (8 个)

| # | 错误 | 症状 | 修复 |
|---|------|------|------|
| 14 | `crate::ops::silu/sigmoid` 不存在 | 编译错误 | 改用 `candle_nn::ops::silu/sigmoid` |
| 15 | `crate::ops::softplus` 不可访问 (gdn 私有) | 编译错误 | 内联: `(ones_like(x) + x.exp()).log()` |
| 16 | `RefCell` 不实现 `Sync` | 编译错误 (GGUFPipeline 要求) | 替换 `Mutex<Option<Tensor>>` + `lock().unwrap()` |
| 17 | `max_scalar()` 方法不存在 | 编译错误 | 改用 `clamp(eps, f64::INFINITY)` |
| 18 | `v_t.sub(&kv)?` 链式 `?` 错误 | 类型推导失败 | 拆分为 `let diff = v_t.sub(&kv)?;` |
| 19 | `GGUFArchitecture` 缺少 `PartialEq` | 编译错误 | 添加 derive |
| 20 | `unimplemented!()` panic | Qwen35MoE per_layer_size 未处理 | 添加 SSM 感知 tensor 查找 |
| 21 | `ffn_norm.weight` 不存在 | "Cannot find tensor info for blk.3.ffn_norm.weight" | `post_attention_norm.weight` 回退 |

### 系统/环境 Bug (2 个)

| # | 错误 | 症状 | 修复 |
|---|------|------|------|
| 22 | GPU 编译 OOM | cargo build 导致 GPU PC 重启 | `CARGO_BUILD_JOBS=1` 单线程 |
| 23 | GPU cargo 不在 PATH (SSH) | "cargo: command not found" | `export PATH=$HOME/.cargo/bin:$PATH` |

---

## 五、Qwen3.6-35B-A3B 架构细节

### 层级结构
```
41 层: [SSM, SSM, SSM, Full-Attn] × 10 + Full-Attn × 1

- 30 SSM (Gated DeltaNet) 层: Conv1d + 门控 Delta 规则循环
- 11 Full Attention 层: RoPE + QK-Norm + GQA (16Q/2KV heads)
- 全部层共享 MoE FFN: 256 experts, top-8 routing, shared expert
```

### SSM (Gated DeltaNet) 关键维度

| 参数 | 值 |
|------|-----|
| d_model | 2048 |
| ssm_inner_size | 4096 |
| ssm_state_size | 128 (状态矩阵维度) |
| n_heads | 32 (值头) |
| n_kv_heads | 16 (键/查询头, 从16重复到32) |
| conv_kernel | 4 |
| full_attention_interval | 4 |

### Gated DeltaNet 循环 (每 token 每头)
```
S_new = alpha * S + k ⊗ delta
where:
  delta = beta * (v - S^T @ k)    // 预测误差
  alpha = exp(gate),  gate < 0     // 衰减因子 ∈ (0,1]
  beta = sigmoid(x @ W_beta)       // 更新强度 ∈ (0,1)
```

---

## 六、跨 Session 常见故障模式

### 故障分类统计

| 类别 | 总次数 | 跨 Session 分布 |
|------|--------|----------------|
| Auto-mode 分类器拦截 | 21 | 全部在本 session (7/24-25) |
| Context 窗口压力 | 29+ | `a7d692f0`(15) + `d8f2e78b`(11) + `ea3d3ea0`(3) |
| SSH 连接失败 | 38+ | `3b612191`(36) + 本 session(2) |
| 工具执行错误 (is_error:true) | 210+ | 跨所有 session |
| 用户中断 | 14+ | 本 session(11) + 其他(3) |
| 系统崩溃 (OOM/WSL) | 2 | GPU OOM(1) + WSL crash(1) |
| Session 截断 | 1 | `057fd8d3` (cron 被终止) |

### 根因时序链

```
上下文窗口有限 (120K token budget, 60% 使用率)
  → 需要频繁 /compact (每 90 分钟)
    → 压缩丢失上下文细节
      → Agent 重做已完成的调查
        → Token 加速消耗
          → 更多 compaction...

独立问题:
  - GPU PC 15GB RAM → cargo build OOM → 重启 → 断开所有连接
  - 手机 SSH 不可达 → 3-machine 方案停滞
  - deepseek-v4-pro 分类器间歇不可用 → 21 次拦截 → 流程被阻断
```

---

## 七、当前资源状态

| 资源 | GPU PC (192.168.1.10) | WSL PC |
|------|----------------------|--------|
| CPU RAM | 15GB (13GB free) | 15GB (12GB free) |
| GPU VRAM | 6GB RTX 4050 (6GB free) | — |
| mistralrs 二进制 | 待重新编译 (SSM 代码已 rsync) | ✅ 98MB release (含 SSM, 7/25 06:10) |
| Qwen3.6-35B-A3B GGUF | 未下载 | ✅ 16GB Q3_K_M |
| PagedAttention KV cache | — | ✅ 4.1GB, 350,720 tokens |

---

## 八、已验证 vs 待完成

| 测试项 | 状态 | 性能 |
|--------|------|------|
| Qwen2-0.5B 跨机桥接 | ✅ 通过 | 197 T/s prompt, 46 T/s decode |
| TCP 协议串行化 | ✅ 通过 | — |
| RemoteConnectionPool 多线程 | ✅ 通过 | — |
| Qwen3.6-35B-A3B GGUF 解析 | ✅ 通过 | 41 层, 248320 vocab |
| SSM 层加载 (blk.0-2) | ✅ 通过 | — |
| SSM 层 forward pass | ⚠️ 编译通过, 未数值验证 | — |
| Attention 层 ffn_norm 回退 | ⚠️ 已修复, 未重新测试 | — |
| Qwen3.6-35B-A3B 端到端推理 | 🚧 **PENDING** | — |
| 跨机桥接 + Qwen3.6-35B-A3B | 🚧 **PENDING** | — |
| GPU CUDA 二进制重建 | 🚧 **PENDING** | 源码已 rsync, 需 CARGO_BUILD_JOBS=1 |
| SSM CUDA kernel 优化 | 🚧 **PENDING** | 目前纯 CPU 逐 token 串行 |

---

## 九、下一步

1. **验证模型加载** — WSL 运行 `mistralrs run -f Qwen_Qwen3.6-35B-A3B-Q3_K_M.gguf -i "Hello"` 验证 41 层全加载和 SSM forward 数值正确
2. **GPU 二进制重建** — `ssh 192.168.1.10 "cd mistral.rs && CARGO_BUILD_JOBS=1 cargo build --release -p mistralrs-cli --features cuda,flash-attn"`
3. **跨机拓扑部署** — GPU 少量 CUDA 层 + WSL CPU 大部份远程层, 分摊 16GB 模型
4. **数值正确性对比** — 纯 CPU vs 桥接输出对比
5. **性能优化** — SSM 纯 CPU 逐 token 是瓶颈, 需要并行化或 CUDA kernel

---

*报告生成: 2026-07-25, 来源: 主 session transcript (5136 行) + 8 个历史 session + git log + memory*
