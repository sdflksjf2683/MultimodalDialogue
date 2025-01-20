import json
import torch
from PIL import Image
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator


#########################################
# 1) 모델/토크나이저 로드 (4bit)
#########################################
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit=True,                    # 4bit quantization
    use_gradient_checkpointing="unsloth", # gradient checkpointing
)

#########################################
# 2) PEFT (LoRA) 설정
#########################################
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,  # 비전 레이어도 미세조정할지
    finetune_language_layers   = True,  # 언어 레이어도 미세조정할지
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,

    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    # target_modules="all-linear", # 필요하다면 지정
)

#########################################
# 3) Dataset 로딩
#    (train_vision.json / valid_vision.json)
#########################################
def load_json_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data는 [{"messages":[...]} ...] 구조
    return data

train_data = load_json_dataset("train_vision.json")
valid_data = load_json_dataset("valid_vision.json")

#########################################
# 4) Trainer 세팅
#########################################
# Vision 모델 학습 시에는 반드시 UnslothVisionDataCollator 사용
data_collator = UnslothVisionDataCollator(model, tokenizer)

# SFTConfig 설정
training_args = SFTConfig(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = 100,   # 혹은 num_train_epochs
    learning_rate = 2e-4,
    fp16 = not is_bf16_supported(),
    bf16 = is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = "vision_outputs",
    report_to = "none",     # or "tensorboard"/"wandb"

    #validation loss를 출력하기 위한 설정 추가 + checkpoint 설정 추가
    evaluation_strategy = "steps",
    eval_steps = 10,
    save_strategy = "steps",
    save_steps = 10,
    save_total_limit = 2,

    # **아래 항목들이 vision finetuning에 필수**
    remove_unused_columns = False,
    dataset_text_field = "",                # text 필드가 따로 없을 때
    dataset_kwargs = {"skip_prepare_dataset": True},
    dataset_num_proc = 4,
    max_seq_length = 2048,
)

#multi-GPU 위한 accelerate 사용
accelerator = Accelerator()
model, data_collator = accelerator.prepare(model, data_collator)

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    data_collator   = data_collator,
    train_dataset   = train_data,
    eval_dataset    = valid_data,
    args            = training_args,
)

#########################################
# 5) 학습 진행
#########################################
FastVisionModel.for_training(model)  # 반드시 호출(훈련 모드 세팅)
trainer_stats = trainer.train()

print("Training finished!")
trainer.save_model("vision_outputs")