from transformers import Trainer, TrainingArguments, AutoProcessor, AutoModelForSeq2SeqLM
# from trl import SFTTrainer
from accelerate import Accelerator
import torch
from config import peft_config, args
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
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=500,
        logging_dir="./logs",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_strategy="epoch",
        report_to="tensorboard",
        gradient_accumulation_steps=8,  # 그라디언트 축적
        fp16=True,  # Mixed Precision
        dataloader_num_workers=4,  # 데이터 로딩 워커 수
        save_total_limit=2,  # 저장할 체크포인트 수 제한
    )

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
