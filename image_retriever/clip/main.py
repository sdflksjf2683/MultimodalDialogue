import os
from PIL import Image
from transformers import AutoProcessor, CLIPModel

#유사도 판단에 사용할 threshold값
THRESHOLD = 0.5


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
        highest_probability = probs[most_similar_index].item()
        
        return {
            "logits": logits_per_image,
            "probabilities": probs,
            "most_similar_index": most_similar_index,
            "higest_probability": highest_probability,
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

#이미지 검색 함수
"""
input: description
output: image path

1. 사진첩 로드
2. 유사도 검사
3. 유사도가 가장 높은 이미지 path 리턴
"""

def image_retriever(description):
    # 이미지 폴더 경로(현재 generated_images 폴더 기준으로 작성)
    directory_path = "./generated_images"

    images, image_paths = load_images_from_directory(directory_path)
    if not images:
        print("Wrong path")
        return
    
    # CLIP initilaize
    clip_processor = CLIPProcessor()

    # trigger sentence로 생성한 image description (테스트에 사용한 description)
    # description = [
    #     "A vibrant selfie taken at a festival, featuring a cheerful person with a big smile. "
    #     "The background is lively, filled with colorful decorations, bright lights, and a crowd of people enjoying the event. "
    #     "The energy of the festival is reflected in their joyful expression and the vibrant atmosphere around them."
    # ]

    # 유사도 계산
    result = clip_processor.calculate_similarity(description, images)
    
    # 가장 높은 유사도 값과 그에 해당하는 이미지 인덱스 가져오기
    highest_probability = result["highest_probability"]
    most_similar_index = result["most_similar_index"]

    # 유사도 값이 기준 이상인 경우 해당 이미지 경로 반환
    if highest_probability >= THRESHOLD:
        most_similar_path = image_paths[most_similar_index]
        print(f"Most similar image path: {most_similar_path}")
        return most_similar_path
    else:
        print("No similar images found above the threshold.")
        return -1

