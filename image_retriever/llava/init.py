import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM

def initialize_model(model_path="4bit/llava-v1.5-13b-3GB"):
    kwargs = {
        "device_map": "auto",
        "load_in_4bit": True,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    }
    # model & tokenizeer initialize
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # vision tower & image processor initialize
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device="cuda")
    image_processor = vision_tower.image_processor

    return model, tokenizer, vision_tower, image_processor
