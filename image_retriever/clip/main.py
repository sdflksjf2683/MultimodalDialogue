import os, logging
from PIL import Image
from transformers import AutoProcessor, CLIPModel
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

#유사도 판단에 사용할 threshold값
THRESHOLD = 0.5


# CLIP 모델 및 프로세서를 관리하는 클래스
class CLIPProcessor:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        logging.info("CLIP model&processor initialized")
    
    #유사도 계산하는 함수
    def calculate_similarity(self, description, images):
        logging.info("Calculating similarity...")
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

#generated_images 폴더에서 이미지 불러오는 함수
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

#flickr30k에서 이미지 불러오는 함수
def load_images_from_flickr30k():
    images = []
    captions = []
    
    dataset = load_dataset("nlphuji/flickr30k", split="test")

    for i, example in enumerate(dataset):
        try:
            # 이미지와 첫 번째 캡션 가져오기
            img = example["image"]  # 이미 `PIL.Image.Image` 객체로 제공됨
            caption = example["caption"][0]  # 첫 번째 캡션만 사용
            images.append(img)
            captions.append(caption)

        except Exception as e:
            logging.error(f"Error loading image {i}: {e}")

    return images, captions
    
    

#이미지 검색 함수
"""
input: description, type(landscape or sth)
output: image path

1. 사진첩 로드
2. type에 따라 적절한 사진첩에서 유사도 검사
3. 유사도가 가장 높은 이미지 path 리턴
"""
def image_retriever(description, type):
    #디렉토리에서 이미지 불러오기
    directory_path = "./generated_images"
    dir_images, dir_image_paths = load_images_from_directory(directory_path)

    #flickr30k에서 이미지 불러오기
    flickr_images, flickr_captions = load_images_from_flickr30k()
    
    # CLIP initilaize
    clip_processor = CLIPProcessor()

    # 인물사진일 경우 directory에서만 가장 높은 유사도 값과 그에 해당하는 이미지 인덱스 가져오기
    dir_result = clip_processor.calculate_similarity(description, dir_images)
    highest_probability = dir_result["highest_probability"]
    most_similar_index = dir_result["most_similar_index"]
    source = "dir"

    if type == "landscape":
        #풍경사진일 경우 flickr30k에서도 검색 후 비교
        flickr_result = clip_processor.calculate_similarity(description, flickr_images)

        f_highest_probability = flickr_result["highest_probability"]
        f_most_similar_index = flickr_result["most_similar_index"]

        # 더 높은 유사도 값을 선택
        if f_highest_probability > highest_probability:
            highest_probability = f_highest_probability
            most_similar_index = f_most_similar_index
            source = "flickr"


    # 유사도 값이 기준 이상인 경우 해당 이미지 경로 반환
    if highest_probability >= THRESHOLD:
        if source == "dir":
            most_similar_path = dir_image_paths[most_similar_index]
        else:
            #flickr의 경우 이미지 저장 후 path 리턴
            most_similar_image = flickr_images[most_similar_index]
            landscape_path = os.path.join(directory_path, "landscapes")
            most_similar_path = os.path.join(landscape_path, f"flickr_{most_similar_index}.png")
            
            # landscapes 디렉토리에 저장(경로는 변경해주세요!)
            try:
                most_similar_image.save(most_similar_path)
                logging.info(f"Flickr image saved to: {most_similar_path}")
            except Exception as e:
                logging.error(f"Error saving Flickr image: {e}")
                return
        
        logging.info(f"Most similar image path: {most_similar_path}")
        return most_similar_path
    else:
        logging.info("No similar images found above the threshold.")
        return -1

