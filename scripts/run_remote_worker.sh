#!/usr/bin/env bash
# 启动 mistral.rs TCP remote worker（在 WSL 或手机上运行）
#
# 用法: ./run_remote_worker.sh [model_file] [listen_addr] [layers]
# 默认: qwen2-0.5b, 0.0.0.0:5051, "0-23"（全部层）
#
# 示例:
#   # WSL: 启动 CPU worker 负责模型全部 24 层
#   ./run_remote_worker.sh
#
#   # WSL: 启动 CPU worker 负责层 8-15
#   ./run_remote_worker.sh "qwen2-0.5b-instruct-q4_0.gguf" "0.0.0.0:5051" "8-15"
#
#   # GPU PC: 启动 CUDA worker（供其他机器卸载层）
#   ./run_remote_worker.sh "Qwen_Qwen3.6-35B-A3B-Q3_K_M.gguf" "0.0.0.0:5050" "0-7"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck source=../config.env
source "${MODE_DIR}/config.env"

MODEL_FILE="${1:-$(basename "$DEFAULT_MODEL")}"
LISTEN_ADDR="${2:-0.0.0.0:${WSL_PORT}}"
LAYERS="${3:-0-0}"  # 0-0 表示不限制层范围，worker 持有全部层

MODEL_PATH="$(dirname "$DEFAULT_MODEL")/${MODEL_FILE}"

echo "=== mistral.rs Remote Worker ==="
echo "Model:        ${MODEL_PATH}"
echo "Listen:       ${LISTEN_ADDR}"
echo "Layers:       ${LAYERS}"
echo "Binary:       ${MISTRALRS_BIN}"
echo "================================"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "$MISTRALRS_BIN" ]; then
    echo "ERROR: mistralrs binary not found: ${MISTRALRS_BIN}"
    echo "Run 'cargo build --release -p mistralrs-cli' first."
    exit 1
fi

exec "${MISTRALRS_BIN}" remote-worker \
    --model-dir "$(dirname "$MODEL_PATH")" \
    --model-file "$MODEL_FILE" \
    --listen "$LISTEN_ADDR" \
    --layers "$LAYERS"
