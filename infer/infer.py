import logging
from PIL import Image
import torch
import json, sys, os
from transformers import (
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
    AutoProcessor,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train')))

from utils import load_images, apply_chat_template_with_dates, collate_fn_

def main():
    model_id = '/home/chanho/Model/photo-sharing/final_Refactorizing/train/output_path/checkpoint-12'

    data_path = '/home/chanho/Model/photo-sharing/final_Refactorizing/dataset/3.make_dataset/test.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        )

    model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config
    )

    processor = AutoProcessor.from_pretrained(model_id)


    data = apply_chat_template_with_dates(data)
    for file in data:
        inputs = collate_fn_([file], processor)
        output = model.generate(**inputs, max_new_tokens=300)
        print("-" * 50)
        print(processor.decode(output[0]))



if __name__ == "__main__":
    main()