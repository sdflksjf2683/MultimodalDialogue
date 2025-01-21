import json
import os
from PIL import Image

def convert_train_json_to_vision(input_path, output_path):
    """
    train.json(위 예시) --> train_vision.json 으로 변환
    """
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)  # 예: [{ "messages": [...], "prompt":..., ... }, ...]

    final_dataset = []
    for item in raw_data:
        # 1) item["messages"]가 아예 없는 경우(또는 빈 리스트)이면 스킵
        if "messages" not in item or not item["messages"]:
            continue

        user_contents = []
        # (A) 전체 대화(messages)를 한 덩어리로 "user" 쪽에 넣기
        lines_for_user_text = []
        for msg in item["messages"]:
            speaker = msg.get("speaker", "Unknown")
            text = msg.get("text", "")
            photo_url = msg.get("photo_url", None)
            photo_desc = msg.get("photo_description", None)

            # 1) text가 있다면, "type":"text" 로 넣을 수도 있고,
            #    대화 흐름을 한 줄씩 쌓고 싶다면 아래처럼 speaker: text 형태로 합치는 방식을 택할 수도 있음
            if text.strip():
                line = f"{speaker}: {text}"
                lines_for_user_text.append(line)

            # 2) photo_url이 있다면, "type":"image" 로 content에 추가
            if photo_url and photo_url.strip():
                # text로 들어갈 "photo_desc"도 있으면 서술
                # 여기선 "desc" 필드로 저장
                user_contents.append({
                    "type": "image",
                    "image": photo_url,
                    "desc": photo_desc
                })

        # (B) messages 내 text들을 하나로 합쳐서 "type":"text" 로 추가
        if lines_for_user_text:
            # 여러 줄을 합쳐서 하나의 text chunk로
            joined_text = "\n".join(lines_for_user_text)
            user_contents.insert(0, {  # 이미지보다 앞에 넣는다면 insert(0, ...)
                "type": "text",
                "text": joined_text
            })

        # 2) "assistant" 쪽 content
        #    - item["utterance"]가 마지막 발화라고 가정
        assistant_utt = item.get("utterance", "")
        assistant_contents = []
        if assistant_utt.strip():
            assistant_contents.append({
                "type": "text",
                "text": assistant_utt
            })
        else:
            # 만약 utterance가 없으면 빈 리스트 -> 나중에 모델이 예측할 문장이 거의 없음
            assistant_contents.append({
                "type": "text",
                "text": ""
            })

        # 3) messages 배열 구성
        conversation = [
            {
                "role": "user",
                "content": user_contents
            },
            {
                "role": "assistant",
                "content": assistant_contents
            }
        ]
        final_dataset.append({"messages": conversation})

    # 4) 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved {len(final_dataset)} samples to {output_path}")

# 사용 예시
if __name__ == "__main__":
    convert_train_json_to_vision(
        input_path="valid.json",
        output_path="valid_vision.json"
    )
