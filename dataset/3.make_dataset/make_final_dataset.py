import os, json, glob
from sklearn.model_selection import train_test_split


def find_full_speaker_name(short_name: str, conversation_item) -> str:
    s1_full = conversation_item["speaker1"]["name"]
    s2_full = conversation_item["speaker2"]["name"]
    s1_first = s1_full.split()[0]
    s2_first = s2_full.split()[0]

    if short_name == s1_first:
        return s1_full
    elif short_name == s2_first:
        return s2_full
    else:
        return short_name

def find_file_by_day(folder_path: str, day_str: str):
    if not os.path.isdir(folder_path):
        return None

    pattern = os.path.join(folder_path, f"*{day_str}*.png")
    candidates = glob.glob(pattern)
    if candidates:
        return candidates[0]
    return None

def add_photo_urls_to_data(data, BASE_DIR):
    """
    기존 add_photo_urls 함수에서
    파일 입출력 부분을 제거하고, data에 직접 photo_url을 추가하는 함수로 변경.
    """
    for conversation in data:
        speaker1_name = conversation["speaker1"]["name"]
        speaker2_name = conversation["speaker2"]["name"]
        pair_folder = f"{speaker1_name}-{speaker2_name}"

        for key in conversation.keys():
            if key.startswith("session_"):
                session_data = conversation[key]
                if "dialogue" in session_data:
                    for turn in session_data["dialogue"]:
                        # photo_description이 있고 photo_url이 없을 경우에만 시도
                        if "photo_description" in turn and turn["photo_description"] and turn.get("photo_url") is None:
                            day = turn.get("day")
                            speaker_short = turn.get("speaker")
                            if not day:
                                continue

                            # speaker_full 찾아서 시도
                            if speaker_short:
                                speaker_full = find_full_speaker_name(speaker_short, conversation)
                                speaker_folder = os.path.join(BASE_DIR, pair_folder, speaker_full)
                                photo_path = find_file_by_day(speaker_folder, day)
                                if photo_path:
                                    turn["photo_url"] = photo_path
                                    continue

                            # 두 명 모두 폴더를 뒤져서 시도
                            for spk_full in [speaker1_name, speaker2_name]:
                                spk_folder = os.path.join(BASE_DIR, pair_folder, spk_full)
                                photo_path = find_file_by_day(spk_folder, day)
                                if photo_path:
                                    turn["photo_url"] = photo_path
                                    break

    return data

def transform_sessions_to_target(dataset):
    results = []

    for top_level_obj in dataset:
        # session_* 키만 추출
        session_keys = [k for k in top_level_obj.keys() if k.startswith("session_")]

        for session_key in session_keys:
            session_data = top_level_obj[session_key]

            # 세션 날짜를 추출해 최종 결과에 추가
            session_date = session_data.get("date", "")

            dialogue = session_data.get("dialogue", [])
            all_dia = session_data.get("all_dia", "")
            messages = []

            for turn in dialogue:
                speaker = turn.get("speaker", "")
                text = turn.get("text", "")
                photo_url = turn.get("photo_url", None)
                photo_description = turn.get("photo_description", None)
                people = turn.get("people", None)
                # 사진에 달린 날짜(day)도 추가
                day = turn.get("day", None)

                msg = {
                    "speaker": speaker,
                    "text": text if text else "",
                    "photo_url": photo_url,
                    "photo_description": photo_description
                }
                # people이 존재할 경우에만 추가
                if people:
                    msg["people"] = people
                # day가 존재할 경우에만 추가
                if day:
                    msg["day"] = day

                messages.append(msg)

            # 마지막 발화 정보
            if dialogue:
                last_turn = dialogue[-1]
                last_speaker = last_turn.get("speaker", "")
                last_text = last_turn.get("text", "")
                if not last_text:
                    last_text = ""
            else:
                last_speaker = ""
                last_text = ""

            session_result = {
                "date": session_date,      # 세션 레벨의 date 추가
                "messages": messages,
                "prompt": all_dia,
                "last_speaker": last_speaker,
                "utterance": last_text
            }
            results.append(session_result)

    return results

def split_and_save_data(data_list, train_file='train.json', valid_file='valid.json', test_file='test.json'):
    # 8:1:1 비율로 분할
    train_data, temp_data = train_test_split(data_list, test_size=0.2, random_state=42)
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # 여기서부터 핵심 로직: test_data의 마지막 발화 text만 ''로 변경
    for item in test_data:
        if item["messages"]:
            # messages의 마지막 index
            last_idx = len(item["messages"]) - 1
            # speaker나 utterance는 변경하지 않고, text만 빈 문자열로
            item["messages"][last_idx]["text"] = ""

    # 변경 후 결과를 파일로 저장
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(valid_file, 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, ensure_ascii=False, indent=4)

    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    
    print(f"Data has been split and saved into {train_file}, {valid_file}, and {test_file}.")


def main(input_json):

    BASE_DIR = "/home/chanho/Model/photo-sharing/Refactorizing/dataset/2.make_image/generated_images" # 이미지 폴더 경로

    # 최종 결과물만 남기는 구조
    dialogue_json = "dialogue.json"

    # 1) 입력 파일 로드
    with open(input_json, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    # 2) data에 직접 photo_url을 추가 (필요하다면)
    data_with_urls = add_photo_urls_to_data(original_data, BASE_DIR)

    # 3) 대화형 변환 (session date, day, people 필드 등 포함)
    all_sessions = transform_sessions_to_target(data_with_urls)

    split_and_save_data(all_sessions)

    # 4) 최종 결과 저장 (dialogue.json)
    with open(dialogue_json, "w", encoding="utf-8") as f:
        json.dump(all_sessions, f, ensure_ascii=False, indent=2)

    print(f"대화형 데이터가 '{dialogue_json}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    input_json = "data.json"  # 입력 JSON 파일 경로
    main(input_json)
