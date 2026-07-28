#!/usr/bin/env bash
# 启动 mistral.rs 桥接 Host（在 GPU PC 上运行）
#
# 用法: ./run_bridge_host.sh <topology.yml> [model_file] [prompt]
#
# 拓扑文件定义每层的设备位置，支持:
#   - cuda[0]: GPU 层
#   - cpu: CPU 层
#   - remote:tcp://IP:PORT: 远端 worker 层
#
# 示例:
#   # GPU PC + WSL 双机桥接（GPU 负责前 8 层，WSL CPU 负责剩余层）
#   ./run_bridge_host.sh topologies/gpu_wsl_bridge.yml
#
#   # 三机桥接（GPU + WSL + 手机）
#   ./run_bridge_host.sh topologies/gpu_wsl_phone_bridge.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck source=../config.env
source "${MODE_DIR}/config.env"

TOPOLOGY="${1:?Usage: $0 <topology.yml> [model_file] [prompt]}"
MODEL_FILE="${2:-$(basename "$DEFAULT_MODEL")}"
PROMPT="${3:-$DEFAULT_PROMPT}"

MODEL_PATH="$(dirname "$DEFAULT_MODEL")/${MODEL_FILE}"

echo "=== mistral.rs Bridge Host ==="
echo "Topology:     ${TOPOLOGY}"
echo "Model:        ${MODEL_PATH}"
echo "Prompt:       ${PROMPT}"
echo "================================"

if [ ! -f "$TOPOLOGY" ]; then
    echo "ERROR: Topology file not found: ${TOPOLOGY}"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "$MISTRALRS_BIN" ]; then
    echo "ERROR: mistralrs binary not found: ${MISTRALRS_BIN}"
    exit 1
fi

# --format gguf 指定 GGUF 格式
# --topology 指定层-设备映射
# -i 交互模式（单次 prompt）
exec "${MISTRALRS_BIN}" run \
    --format gguf \
    --topology "$TOPOLOGY" \
    --model-id "$MODEL_PATH" \
    --max-seq-len 4096 \
    -i "$PROMPT"
