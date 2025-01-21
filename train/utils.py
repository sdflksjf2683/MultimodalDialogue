import json
from transformers import (
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
    AutoProcessor,
)

from PIL import Image

def load_images(image_paths):
    """
    이미지 파일 경로 리스트를 받아 PIL.Image 객체 리스트를 반환합니다.
    """
    loaded_images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")  # RGB로 변환하여 불러오기
            loaded_images.append(img)
        except Exception as e:
            print(f"Failed to load image {path}: {e}")
    return loaded_images



def load_json_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def collate_fn_(examples, processor):
    # 텍스트와 이미지를 분리
    texts = [processor.apply_chat_template(example["messages"], tokenize=False).strip("Anytime, Kendra. Keep dreaming big and let your creativity flow. I'm here cheering you on from the West Coast!") for example in examples]
    image_paths = [example["image"] for example in examples]

    # 이미지 로드
    loaded_images = [load_images(paths) for paths in image_paths]

    # processor로 텍스트 및 이미지 처리
    batch = processor(text=texts, images=loaded_images, return_tensors="pt", padding=True)

    return batch



def apply_chat_template_with_dates(conversations):   

    SPECIAL_PROMPT_TEMPLATE = """You are a conversational system that can share excellent images and  generate responses based on the given context or create descriptions for images. When you share a image, you can generate <IMG> tokens with the following image description.  The image description must be provided in the following format:
<IMG>[Image by speaker | landscape or people | Image description: {{Image description}}]<IMG>.
The following conversation takes place between {0} and {1} on {2}.
The provided images are photos shared within the conversation below. Make sure to carefully check the date and description of the images."""



    SPECIAL_PROMPT_TEMPLATE = """You are a conversational system that generate responses based on the given context. The following conversation takes place between {0} and {1} on {2}.
The provided images are photos shared within the conversation below. Make sure to carefully check the date and description of the images."""

    results = []

    for conv in conversations:
        all_messages = conv.get("messages", [])
        last_speaker = conv.get("last_speaker", "Unknown")
        conv_date = conv.get("date", "Unknown date")

        # 메시지가 없으면 최소 구조
        if not all_messages:
            speakers = ["SpeakerA", "SpeakerB"]
            special_prompt = SPECIAL_PROMPT_TEMPLATE.format(speakers[0], speakers[1], conv_date)
            results.append({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": special_prompt}
                        ]
                    }
                ],
                "image": []
            })
            continue

        # 화자(스피커) 추출
        speakers = list({msg.get("speaker", "Unknown") for msg in all_messages})
        if len(speakers) < 2:
            speakers = (speakers + ["Unknown", "Unknown"])[:2]
        else:
            speakers = speakers[:2]

        # 템플릿 (시작 프롬프트)
        special_prompt = SPECIAL_PROMPT_TEMPLATE.format(speakers[0], speakers[1], conv_date)

        # ─────────────────────────────────────────────────────────
        # 1) user 메시지 (마지막 메시지 제외) 구성
        # ─────────────────────────────────────────────────────────
        user_side_messages = all_messages[:-1]
        user_contents = []
        accumulated_text = []

        def flush_accumulated_text():
            """누적된 텍스트가 있으면 하나의 text 블록으로 user_contents에 추가"""
            if accumulated_text:
                joined = "\n".join(accumulated_text)
                user_contents.append({"type": "text", "text": joined})
                accumulated_text.clear()

        # 맨 처음에 prompt 추가
        accumulated_text.append(special_prompt)

        for msg in user_side_messages:
            spk = msg.get("speaker", "Unknown")
            txt = msg.get("text", "")
            purl = msg.get("photo_url")
            pdesc = msg.get("photo_description")
            ppl = msg.get("people")
            pday = msg.get("day")

            # 텍스트가 있다면 누적
            if txt:
                accumulated_text.append(f"{spk}: {txt}")

            # 이미지가 있다면 flush 후 이미지 블록 추가
            if purl:
                flush_accumulated_text()
                user_contents.append({
                    "type": "image",
                    "image": purl,
                    "description": pdesc,
                    "people": ppl,
                    "day": pday
                })

        # 누적된 텍스트가 남아있으면 flush
        flush_accumulated_text()

        user_message = {"role": "user", "content": user_contents}

        # ─────────────────────────────────────────────────────────
        # 2) 마지막 메시지와 utterance 확인
        # ─────────────────────────────────────────────────────────
        last_msg = all_messages[-1]
        # 우선 conv["utterance"]를 사용, 없으면 last_msg["text"] 사용
        last_text = conv.get("utterance", last_msg.get("text", ""))
        last_photo_url = last_msg.get("photo_url")
        last_photo_desc = last_msg.get("photo_description")
        last_people = last_msg.get("people")
        last_day = last_msg.get("day")

        # 마지막 메시지가 text='' + image 없음인지 체크
        is_last_msg_empty_text = (last_msg.get("text", "") == "")
        is_last_msg_no_image = (last_photo_url is None)

        # ─────────────────────────────────────────────────────────
        # 3) 마지막 메시지가 "text='' + photo_url 없음" → 특수 처리
        #    => "assistant" 메시지 제거,
        #       user의 마지막 텍스트에 "\n마지막화자:" 만 추가,
        #       last_text 내용은 붙이지 않음
        # ─────────────────────────────────────────────────────────
        if is_last_msg_empty_text and is_last_msg_no_image:
            # "assistant" 메시지 없이, 마지막에 "마지막 화자:"만 붙이기
            if user_contents and user_contents[-1]["type"] == "text":
                # user_contents[-1]["text"]에 '\nMarcus:' 만 붙이기
                user_contents[-1]["text"] += f"\n{last_speaker}:"
            else:
                # 혹시 마지막 블록이 이미지일 경우 등
                user_contents.append({
                    "type": "text",
                    "text": f"{last_speaker}:"
                })

            # 어시스턴트 메시지 없이 종료
            session_messages = [user_message]
        else:
            # ─────────────────────────────────────────────────────
            # 4) 기존 로직: 마지막 메시지를 assistant로 처리
            # ─────────────────────────────────────────────────────
            assistant_contents = []
            if last_text:
                assistant_contents.append({
                    "type": "text",
                    "text": f"{last_speaker}: {last_text}"
                })
            if last_photo_url:
                assistant_contents.append({
                    "type": "image",
                    "image": last_photo_url,
                    "description": last_photo_desc,
                    "people": last_people,
                    "day": last_day
                })

            assistant_message = {"role": "assistant", "content": assistant_contents}
            session_messages = [user_message, assistant_message]

        # ─────────────────────────────────────────────────────────
        # 5) 최종 구조
        # ─────────────────────────────────────────────────────────
        session_dict = {
            "messages": session_messages
        }

        # 이미지 목록 정리
        collected_images = []
        for msg in all_messages:
            if msg.get("photo_url"):
                collected_images.append(msg["photo_url"])
        session_dict["image"] = collected_images

        results.append(session_dict)

    return results

