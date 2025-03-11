#!/bin/bash

# Set GPU device to use GPU 5
export CUDA_VISIBLE_DEVICES=5

MODEL_PATH="chano12/photo_sharing_response_generation"
MODEL_BASE="Qwen/Qwen2.5-VL-7B-Instruct"
TEST_JSON="/home/chanho/Model/photo-sharing/final_Refactorizing_2/dataset/test.json"


python infer.py \
  --model-path "$MODEL_PATH" \
  --model-base "$MODEL_BASE" \
  --test-json "$TEST_JSON" \
  $DEBUG
