import argparse, json
import torch
from PIL import Image
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
    AutoProcessor,
)

from utils import load_json_dataset, load_images, apply_chat_template_with_dates

def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi-modal dialogue model with WandB logging.")

    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="배치 사이즈")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient Accumulation Steps")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="학습 epoch 수")
    parser.add_argument("--max_steps", type=int, default=-1, help="학습 스텝(에폭 대신 스텝을 쓰려면 설정. -1이면 무시)")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup 스텝")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=1, help="로그 표시 스텝 간격")
    parser.add_argument("--output_dir", type=str, default="vision_outputs", help="모델 저장 경로")
    parser.add_argument("--seed", type=int, default=3407, help="시드(SEED)")

    parser.add_argument("--train_data_path", type=str, default="train_vision.json", help="학습 데이터 경로")
    parser.add_argument("--valid_data_path", type=str, default="valid_vision.json", help="검증 데이터 경로")

    return parser.parse_args()

def main():
    args = parse_args()
    model_id = "unsloth/Llama-3.2-11B-Vision-Instruct"

    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit=True,                    # 4bit quantization
        use_gradient_checkpointing="unsloth", # gradient checkpointing
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r           = 16,
        lora_alpha  = 16,
        lora_dropout= 0.0,
        bias        = "none",
        random_state= args.seed,
        use_rslora  = False,
        loftq_config= None,
    )

    train_data = apply_chat_template_with_dates(load_json_dataset(args.train_data_path))
    valid_data = apply_chat_template_with_dates(load_json_dataset(args.valid_data_path))

    processor = AutoProcessor.from_pretrained(model_id)


    def collate_fn(examples):
        # 텍스트와 이미지를 분리
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        image_paths = [example["image"] for example in examples]

        # 이미지 로드
        loaded_images = [load_images(paths) for paths in image_paths]

        # processor로 텍스트 및 이미지 처리
        batch = processor(text=texts, images=loaded_images, return_tensors="pt", padding=True)

        # 라벨 처리
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch


    training_args = SFTConfig(
        per_device_train_batch_size  = args.per_device_train_batch_size,
        gradient_accumulation_steps  = args.gradient_accumulation_steps,
        num_train_epochs             = args.num_train_epochs if args.max_steps < 0 else 1,
        max_steps                    = args.max_steps if args.max_steps > 0 else -1,
        warmup_steps                 = args.warmup_steps,
        learning_rate                = args.learning_rate,
        fp16                         = not is_bf16_supported(),
        bf16                         = is_bf16_supported(),
        logging_steps                = args.logging_steps,
        optim                        = "adamw_8bit",
        weight_decay                 = 0.01,
        lr_scheduler_type            = "linear",
        seed                         = args.seed,
        output_dir                   = args.output_dir,

        report_to                    = "wandb",

        remove_unused_columns        = False,
        dataset_text_field           = "",  
        dataset_kwargs               = {"skip_prepare_dataset": True},
        dataset_num_proc             = 4,
        max_seq_length               = 1024,
        save_steps = 1000
    )

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = processor.tokenizer,
        data_collator = collate_fn,
        train_dataset = train_data,
        eval_dataset  = valid_data,
        args          = training_args,
    )

    FastVisionModel.for_training(model)  
    trainer.train()

    model.save_pretrained("lora_model") 
    tokenizer.save_pretrained("lora_model")

    print("Training finished!")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
