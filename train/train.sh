#!/bin/bash

RESULT_DIR="result"
CURRENT_TIME=$(TZ=Asia/Seoul date +"%Y-%m-%d-%H.%M.%S")
LOG_DIR="${RESULT_DIR}/${CURRENT_TIME}"
mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/train.log"

RESULT_DIR="output_path"
CURRENT_TIME=$(TZ=Asia/Seoul date +"%Y-%m-%d-%H.%M.%S")
OUTPUT_DIR="${RESULT_DIR}/${CURRENT_TIME}"
mkdir -p "$OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES=5

python3 main.py \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 3e-4 \
  --warmup_steps 10 \
  --train_data_path "/home/chanho/Model/photo-sharing/final_Refactorizing/dataset/3.make_dataset/train.json" \
  --valid_data_path "/home/chanho/Model/photo-sharing/final_Refactorizing/dataset/3.make_dataset/valid.json" \
  --output_dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOGFILE"

echo "Training finished!"
echo "Log file saved at: $LOGFILE"
