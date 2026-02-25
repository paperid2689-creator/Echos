#!/usr/bin/env bash
set -euo pipefail

ROOT="/Images Dataset"
SCRIPT_QWEN25="$ROOT/MICCAI2026/finetune/finetune_qwen25_lora.py"
SCRIPT_QWEN3="$ROOT/MICCAI2026/finetune/finetune_qwen3_lora.py"
LOG_DIR="$ROOT/MICCAI2026/finetune/logs"
mkdir -p "$LOG_DIR"

source /conda.sh

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200

CUDA_VISIBLE_DEVICES=0,1,2 conda run --no-capture-output -n base torchrun --nproc_per_node=3 --master_port=29510 "$SCRIPT_QWEN25" \
  --model-path "/llms/Qwen2.5-VL-7B/" \
  --dataset-root "$ROOT" \
  --output-root "$ROOT/MICCAI2026/finetune/runs" \
  --train-ratio 0.7 \
  --max-train-samples 0 \
  --test-fraction 1.0 \
  --eval-max-samples 1000 \
  --epochs 50 \
  --batch-size 6 \
  --image-size 224 \
  --cache-samples 10 \
  --fraction-per-class 1.0 \
  > "$LOG_DIR/finetune_gpu012_qwen25.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3,4,5 conda run --no-capture-output -n base torchrun --nproc_per_node=3 --master_port=29511 "$SCRIPT_QWEN3" \
  --model-path "/llms/Qwen3-VL-8B-Instruct/" \
  --dataset-root "$ROOT" \
  --output-root "$ROOT/MICCAI2026/finetune/runs" \
  --train-ratio 0.7 \
  --max-train-samples 0 \
  --test-fraction 1.0 \
  --eval-max-samples 1000 \
  --epochs 50 \
  --batch-size 6 \
  --image-size 224 \
  --cache-samples 10 \
  --fraction-per-class 1.0 \
  > "$LOG_DIR/finetune_gpu345_qwen3.log" 2>&1 &

echo "Launched 2 LoRA fine-tune jobs."
echo "Logs: $LOG_DIR"
