import argparse, json, os, warnings
import torch
from utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

def replace_image_tokens(input_string):
    return input_string.replace("<image>", "<|image_start|><image><|image_end|>")
def llava_to_openai(conversations):
    role_mapping = {"human": "user", "gpt": "assistant"}
    transformed_data = []
    for conv in conversations:
        from_role = role_mapping.get(conv["from"], conv["from"])
        new_content = replace_image_tokens(conv["value"])
        transformed_data.append({
            "role": from_role,
            "content": new_content
        })
    return transformed_data
def gather_image_files(sample):
    if "image" not in sample or not sample["image"]:
        return []
    return sample["image"]

def build_prompt_and_inputs(sample, processor, device):
    image_files = gather_image_files(sample)
    raw_convs = sample.get("conversations", [])
    transformed_convs = llava_to_openai(raw_convs)


    prompt_str = processor.apply_chat_template(
        transformed_convs,
        tokenize=False,
        add_generation_prompt=True  # 학습 시와 동일
    )
    SYSTEM_MESSAGE = """You will be provided with a conversation between two people and images. Your task is to generate the next response based on the given dialogue history. There are two possible types of responses you can generate:
Text: The response should be logical and consistent with the given context.
Image: When it is more appropriate to send an image, include the token <IMG> in the response.
When sending an image, ensure the image description follows this format:
<IMG>[Image by {speaker} | {people} | Image description: {Image description} | date: {date}]<IMG>
And when generating a text response, if the next response is expected to include an image, you must additionally include the token [DST] in the text.
The provided images are photos shared within the conversation below. Be sure to carefully review the dates and descriptions of the images before generating your response.
Dialogue History:"""
    prompt = prompt_str.replace("You are a helpful assistant.", SYSTEM_MESSAGE) 
    prompt_str = prompt
    image_inputs, video_inputs = process_vision_info(transformed_convs)

    # E) processor(...) => 최종 inputs
    inputs = processor(
        text=[prompt_str],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    return inputs, prompt_str


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    disable_torch_init()

    # 1) 모델 + Processor 로드
    use_flash_attn = True
    model_name = get_model_name_from_path(args.model_path)
    processor, model = load_pretrained_model(
        model_base=args.model_base,
        model_path=args.model_path,
        model_name=model_name,
        device_map=device,
        device=device,
        use_flash_attn=use_flash_attn
    )
    model.eval()

    # 2) test dataset(JSON) 로드
    with open(args.test_json, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    print(f"Loaded {len(data_list)} samples from {args.test_json}")

    # 3) generation config
    generation_args = {
        "max_new_tokens": 512,
        "temperature": 0.9,
        "do_sample": (0.9 > 0),
        "repetition_penalty": 1.0,
        "eos_token_id": processor.tokenizer.eos_token_id,
        'num_beams': 3,
    }

    results = []

    for idx, sample in enumerate(data_list):

        inputs, prompt_str = build_prompt_and_inputs(sample, processor, device)
        print(prompt_str)
        print("-------------------")
        # C) 모델 generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_args)
        output_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {output_text}")
        assert False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default = 'chano12/photo_sharing_response_generation',
                        help="Path to merged Qwen2-VL model. (LoRA merged if needed.)")
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Used if model-path is not merged.")
    parser.add_argument("--test-json", type=str, default='/home/chanho/Model/photo-sharing/final_Refactorizing_2/dataset/test.json',
                        help="Your test dataset JSON. (same structure as training dataset).")
    args = parser.parse_args()
    main(args)
