#!/bin/bash

set_gpu() {
  echo "Select GPU: 2 or 5"
  read -p "Enter GPU number: " gpu
  if [[ "$gpu" == "2" || "$gpu" == "5" ]]; then
    export CUDA_VISIBLE_DEVICES=$gpu
    echo "Using GPU $gpu"
  else
    echo "Invalid GPU selection. Defaulting to GPU 2."
    export CUDA_VISIBLE_DEVICES=2
  fi
}

set_gpu

MODEL_NAME='gpt-4o'
API_KEY=''
DATA_PATH='../data_10.json'
python3 make_album.py --model_name $MODEL_NAME --api_key $API_KEY --data_path $DATA_PATH