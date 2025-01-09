from transformers import AutoProcessor
import json

"""
데이터셋 그대로 쓸거면 굳이 전처리 안해도 될 것 같아요

def format_data(sample):
    #관계
    relationship = f"The characters have the following relationship: {sample['relationship']}."

    # 대화 내용
    user_texts = [dialogue["text"] for dialogue in sample["dialogue"]]
    photo_descriptions = [dialogue.get("photo_description", "") for dialogue in sample["dialogue"]]
    photo_urls = [dialogue.get("photo_url", None) for dialogue in sample["dialogue"]]

    # 대화 내용 user_content에 합치기
    user_content = []
    for text, photo_desc, photo_url in zip(user_texts, photo_descriptions, photo_urls):
        content = [{"type": "text", "text": text}]
        if photo_desc and photo_url:
            content.append({"type": "image", "image": photo_url})
        user_content.append(content)

    # summary
    summary = [{"type": "text", "text": sample["summary"]}]

    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": relationship}],
            },
            {
                "role": "user",
                "content": user_content,  #
            },
            {
                "role": "assistant",
                "content": summary,  
            },
        ],
    }
"""

# 데이터셋 로드+전처리
def load_and_prepare_data():
    file_path = "path/to/your/data.json"

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    #데이터 그대로 쓸거면 전처리는 필요 없습니다ㅏ
    # formatted_data = [format_data(sample) for sample in raw_data]
    return raw_data

# 데이터 전처리+배치
def collate_data(examples, processor):
    # texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    # image_inputs = [] #이미지 input

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  # 패딩 토큰 무시
    batch["labels"] = labels

    return batch
