from transformers import Trainer, AutoProcessor, AutoModelForSeq2SeqLM
# from accelerate import Accelerator
import torch
from config import training_args
from data import load_and_prepare_data, collate_data

# 모델 및 프로세서 로드
def load_model_and_processor(model_id):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto", #multi gpu 설정 
    )
    return model, processor

# 학습 실행
def train_model():
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # 데이터셋 로드 및 전처리
    dataset = load_and_prepare_data()

    # 모델 및 프로세서 로드
    model, processor = load_model_and_processor(model_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda examples: collate_data(examples, processor),
        tokenizer=processor.tokenizer,
    )


    #학습하고 저장
    trainer.train()
    trainer.save_model("./results")

if __name__ == "__main__":
    train_model()
