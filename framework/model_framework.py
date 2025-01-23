import os, json, sys
from transformers import AutoModelForVision2Seq, AutoProcessor


#각 파일에서 필요한 함수 호출
from train.main import main as train_model
from infer.infer_modify import generate_responses
from image_retriever.clip.main import image_retriever
from train.utils import apply_chat_template_with_dates, load_json_dataset

IMAGE_UTILS_PATH = os.path.join(os.path.abspath(__file__), '../dataset/2.make_image')
sys.path.append(IMAGE_UTILS_PATH)
#일단 다 가져오긴 했는데, 이거 Image_utils에서 조건문으로 처리해주는 게 더 깔끔할 것 같긴 해요..!
from Image_utils import (
    generate_uniportrait_image,
    generate_uniportrait_image_with_2_faces,
    generate_uniportrait_image_with_3_faces,
    generate_uniportrait_image_with_4_faces,
    generate_uniportrait_image_with_5_faces,
    generate_landscape_image,
    generate_cosmicman_image,
)


#학습
def train():
    train_model()

#모델로드(학습 코드와 연결해도 무방)
def load_model_and_processor(model_path):
    #아래는 url로 가져오는 버전. 모델 id 가져오는 방법 수정 필요
    model = AutoModelForVision2Seq.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    return model, processor

#response 생성
def generate_response(model_id, processor, test_data_path):
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    session_history = []
    max_turns = 15 #최대 턴 수(조정가능)

    for turn in range(max_turns): 

        context = " ".join([entry["text"] for entry in session_history[-5:]]) #context는 임의로 최근 5개 발화만 가져왔는데 수정핻도 됩니다

        generated_text = generate_responses(test_data, model_id)

        #image response일 경우
        if "<IMG>" in generated_text:

            trigger_sentence = "" #이걸 감지하는지 몰라 일단 제외했습니다.
            description = "" #trigger senetenc로 생성된 description. 모델이 output으로 만들어주는지 확인 필요

            #사진첩에서 이미지 검색
            this_turn_response = image_retriever(description)

            if this_turn_response == -1: #검색에 실패한 경우 이미지 생성
                #여기서 사람이 몇 명인지에 따라 함수를 다르게 호출하는데 조건문이 너무 길어집니다.
                generate_landscape_image(
                    prompt=description,
                    negative_prompt="blurry, low quality",
                    output_filename="landscape.png"
                )
        
        #text response일 경우
        else:
            this_turn_response = generated_text
        
        session_history.append({"text": this_turn_response})

        #종료 조건 있으면 추가
        #break
    
    return session_history


#데이터셋 생성 후 모델 업데이트
# def update_model(new_data_path):
    #데이터셋 생성 함수 호출

    #new data로 모델 훈련 코드 실행(훈련 코드에서 기존 모델 로드한 후 추가 데이터로 학습하게 변경)
    #try:
        #모델 훈련
    #except Exception as e:
        #에러 로그 출력

#메인함수
def main():
    model_path = "" #모델 불러올 경로(id)
    test_data_path = "" #기존 데이터 불러올 경로
    new_data_path = "" #새로 만든 데이터 저장할 경로

    #1. 모델 훈련
    # train()

    #2. 훈련한 모델 및 프로세서 호출
    model_id, processor = load_model_and_processor(model_path)

    ##############ongoing session#############
    #3. 응답 생성
    session_history = generate_response(model_id, processor, test_data_path)

    ##############end session#############
    #4. 데이터셋 생성 후 모델 업데이트
    # update_model(session_history, new_data_path)

if __name__ == "__main__":
    main()