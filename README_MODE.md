# mistral.rs TCP 桥接模式：GPU + CPU 分层推理

本目录记录使用 **mistral.rs TCP 桥接**（`remote:tcp://` 拓扑 + `DeviceMapper` + `RemoteLayerMapper`）实现 GPU / CPU / 手机跨机异构推理的全过程。

- 状态：⚠️ Qwen2-0.5B 跨机桥接已验证通过；Qwen3.6-35B-A3B 端到端推理未完成（GPU PC OOM 崩溃中断）
- 框架：mistral.rs v0.9.0-dev（基于 EricLBuehler/mistral.rs，16 个自定义 bridge commits）
- 模型：`Qwen2-0.5B-Instruct-Q4_0`、`Qwen3.6-35B-A3B-Q3_K_M`（目标）
- 设备：GPU PC（RTX 4050 6GB）+ WSL（CPU 15GB）+ Mate 40 Pro（CPU 8GB，未实际参与）

## 目录

- `config.env` — 节点地址、模型路径、拓扑参数
- `scripts/` — remote worker、bridge host、GPU 编译脚本
- `topologies/` — 双机/三机桥接 YAML 拓扑文件
- `docs/` — 源码更改报告与通宵 session 报告
- `logs/` — 运行日志
- `README_MODE.md` — 本文件（模式说明）

## 与 llama.cpp RPC 的关键区别

| 特性 | mistral.rs Bridge | llama.cpp RPC |
|------|-------------------|---------------|
| 通信协议 | 自定义 TCP 二进制协议 | gRPC |
| 层粒度 | 任意连续层范围 | 从第 N 层开始卸载 |
| 远端设备语法 | `remote:tcp://IP:PORT` (YAML) | `--rpc IP:PORT` (CLI) |
| 手机支持 | ⚠️ 需手机编译 mistralrs worker | ✅ Termux arm64 编译 |
| Qwen3.6-35B-A3B SSM | ✅ 已实现（代码层） | ❌ 不支持 |
| 生产就绪 | ❌ 实验阶段 | ✅ 可用 |

## 快速开始

### 1. 编译 WSL 本机二进制

```bash
cd /home/atituiset/Projects/gpu-cpu-phone-test/mistral.rs
cargo build --release -p mistralrs-cli
```

### 2. 启动 WSL Remote Worker（终端 1）

```bash
cd /home/atituiset/Projects/gpu-cpu-phone-test/mistral.rs
./scripts/run_remote_worker.sh
```

### 3. 启动桥接 Host 推理（终端 2）

```bash
cd /home/atituiset/Projects/gpu-cpu-phone-test/mistral.rs
./scripts/run_bridge_host.sh topologies/gpu_wsl_bridge.yml
```

### 4. 编译 GPU PC CUDA 二进制（可选）

```bash
./scripts/build_gpu_binary.sh 1
```

## 已验证 vs 待完成

| 测试项 | 状态 | 性能 |
|--------|------|------|
| Qwen2-0.5B 跨机桥接 | ✅ 通过 | 197 T/s prompt, 46 T/s decode |
| TCP 协议串行化 | ✅ 通过 | — |
| RemoteConnectionPool 多线程 | ✅ 通过 | — |
| Qwen3.6-35B-A3B GGUF 解析 | ✅ 通过 | 41 层, 248320 vocab |
| SSM 层加载 (blk.0-2) | ✅ 通过 | — |
| SSM 层 forward pass | ⚠️ 编译通过, 未数值验证 | — |
| Attention 层 ffn_norm 回退 | ⚠️ 已修复, 未重新测试 | — |
| Qwen3.6-35B-A3B 端到端推理 | 🚧 PENDING | — |
| 跨机桥接 + Qwen3.6-35B-A3B | 🚧 PENDING | — |
| GPU CUDA 二进制重建 | 🚧 PENDING | 源码已 rsync, 需 CARGO_BUILD_JOBS=1 |
| SSM CUDA kernel 优化 | 🚧 PENDING | 目前纯 CPU 逐 token 串行 |

## 关键结论

mistral.rs TCP 桥接在**小模型（Qwen2-0.5B）上跨机推理可行**，性能与 llama.cpp RPC 同量级。但**目标模型 Qwen3.6-35B-A3B（16GB GGUF）端到端推理未跑通**——最后阶段 GPU PC 因 `cargo build --release --features cuda` 并行编译耗尽 15GB RAM 导致 kernel OOM 重启，WSL 也崩溃。

核心遗留问题：
1. GPU PC 编译 OOM — 需 `CARGO_BUILD_JOBS=1` 单线程编译
2. Qwen3.6-35B-A3B 的 SSM (Gated DeltaNet) 层 forward pass 数值正确性未验证
3. 手机端未实际参与 mistral.rs 桥接测试

> 详细分析见 `docs/report.md`（源码更改报告）和 `docs/overnight-session.md`（通宵 session 记录）。

## 相关源码变更

16 个 commits（`e527fbeb7` → `f66ff28c2`，2026-07-24/25），涵盖：
- `RemoteLayerMapper` + `RemoteConnectionPool`（新文件 `device_map/remote.rs`）
- 7 个 GGUF 模型文件的 layers→Option 适配
- `forward_from_layer()` + `reset_kv_cache()` Pipeline trait 扩展
- TCP remote worker 守护进程
- 8 个桥接协议 bug 修复 + 8 个 SSM 实现 bug 修复

完整源码 diff 见 `docs/report.md`。

## 架构简图

```
┌─────────────────────────┐
│ GPU PC (RTX 4050 6GB)   │
│ mistralrs run --topology│
│ Layers 0-7: cuda[0]     │ ──TCP── ┌──────────────────────┐
│                         │         │ WSL (CPU 15GB)        │
└─────────────────────────┘         │ mistralrs remote-worker│
                                    │ Layers 8-23: CPU      │
                                    └──────────────────────┘
                                              │
                                    ┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
                                    │ Mate 40 Pro (8GB)     │
                                    │ (已配置, 未实际测试)   │
                                    └─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```
