#!/usr/bin/env bash
# Example launcher for PAAC. Adjust the arguments to match your experiment.

set -euo pipefail

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "[error] GEMINI_API_KEY is not set. export it before running."
    exit 1
fi

# Optional: comma-separated list of NVIDIA NIM keys for hosted on-device models.
# export API_KEYS="key1,key2,key3"

# Optional: vLLM serving port for self-hosted on-device models.
export VLLM_PORT="${VLLM_PORT:-8000}"

DATASET="${DATASET:-gaia}"
PRIVACY_LEVEL="${PRIVACY_LEVEL:-3}"
STRATEGY="${STRATEGY:-parallel_plan_and_solve}"
DECISION="${DECISION:-joint}"
TIER="${TIER:-base}"
DEVICE_MODEL="${DEVICE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
CLOUD_MODEL="${CLOUD_MODEL:-gemini-3-flash-preview}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
WORKERS="${WORKERS:-20}"
RUN_NAME="${RUN_NAME:-default}"

python src/main.py \
    --dataset "${DATASET}" \
    --privacy_level "${PRIVACY_LEVEL}" \
    --strategy "${STRATEGY}" \
    --decision_making "${DECISION}" \
    --tier "${TIER}" \
    --device_model "${DEVICE_MODEL}" \
    --cloud_model "${CLOUD_MODEL}" \
    --num_samples "${NUM_SAMPLES}" \
    --workers "${WORKERS}" \
    --run_name "${RUN_NAME}" \
    --vllm_port "${VLLM_PORT}"
