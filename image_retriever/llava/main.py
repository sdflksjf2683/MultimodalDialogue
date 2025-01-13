import os
from PIL import Image
import torch
from torch.nn.functional import cosine_similarity
from init import initialize_model  # init.py의 initialize_model

# 이미지 불러오기
def load_images_from_directory(directory_path):
    images, image_paths = [], []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                try:
                    img = Image.open(full_path).convert("RGB")
                    images.append(img)
                    image_paths.append(full_path)
                except Exception as e:
                    print(f"Error loading image {full_path}: {e}")
    return images, image_paths

# 이미지 임베딩 생성
def get_image_embeddings(images, vision_tower, image_processor):
    image_embeddings = []
    for img in images:
        processed = image_processor.preprocess(img, return_tensors="pt")
        pixel_values = processed["pixel_values"].to("cuda")
        with torch.no_grad():
            embedding = vision_tower(pixel_values)
        image_embeddings.append(embedding)
    return torch.cat(image_embeddings)

# 텍스트 임베딩 생성
def get_text_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        embedding = model.get_input_embeddings()(inputs.input_ids).mean(dim=1)
    return embedding


def main():
    # config
    model_path = "4bit/llava-v1.5-13b-3GB"
    directory_path = "/content/generated_images"

    #trigger sentence로 만든 description
    description = ""

    #initialize
    model, tokenizer, vision_tower, image_processor = initialize_model(model_path)

    # 이미지 로드
    images, image_paths = load_images_from_directory(directory_path)
    if not images:
        print("Wrong path")
        return

    image_embeddings = get_image_embeddings(images, vision_tower, image_processor)
    text_embedding = get_text_embedding(description, model, tokenizer).to(torch.float32)

    similarities = cosine_similarity(image_embeddings.mean(dim=1), text_embedding.mean(dim=1))

    #마찬가지로 일단 image path만 출력
    most_similar_index = similarities.argmax().item()
    print(f"Most similar image path: {image_paths[most_similar_index]}")

if __name__ == "__main__":
    main()
