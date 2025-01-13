import os
from PIL import Image
from transformers import AutoProcessor, CLIPModel


# CLIP 모델 및 프로세서를 관리하는 클래스
class CLIPProcessor:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    #유사도 계산하는 함수
    def calculate_similarity(self, description, images):
        # Prepare inputs
        inputs = self.processor(
            text=description,
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=0)
        
        most_similar_index = probs.argmax().item()
        
        return {
            "logits": logits_per_image,
            "probabilities": probs,
            "most_similar_index": most_similar_index
        }

#이미지 불러오는 함수
def load_images_from_directory(directory_path):
    images = []
    image_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png')):
                full_path = os.path.join(root, file)
                try:
                    img = Image.open(full_path).convert("RGB")
                    images.append(img)
                    image_paths.append(full_path)
                except Exception as e:
                    print(f"Error loading image {full_path}: {e}")
    return images, image_paths


def main():
    # 이미지 폴더 경로(현재 generated_images 폴더 기준으로 작성)
    directory_path = "./generated_images"

    images, image_paths = load_images_from_directory(directory_path)
    if not images:
        print("Wrong path")
        return
    
    # CLIP initilaize
    clip_processor = CLIPProcessor()

    # trigger sentence로 생성한 image description (테스트에 사용한 description)
    description = [
        "A vibrant selfie taken at a festival, featuring a cheerful person with a big smile. "
        "The background is lively, filled with colorful decorations, bright lights, and a crowd of people enjoying the event. "
        "The energy of the festival is reflected in their joyful expression and the vibrant atmosphere around them."
    ]

    # 유사도 계산
    result = clip_processor.calculate_similarity(description, images)
    
    # 결과 출력(일단은 이미지 path만 출력하도록 했습니다! output을 어떻게 줘야 할지 모르겠어서..)
    most_similar_index = result["most_similar_index"]
    most_similar_path = image_paths[most_similar_index]
    print("Most similar image path:", most_similar_path)


if __name__ == "__main__":
    main()
