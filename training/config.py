from peft import LoraConfig
from trl import SFTConfig

# LoRA 설정
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # LLaMA3 타겟 모듈
    task_type="CAUSAL_LM",
)

# SFT 설정
args = SFTConfig(
    output_dir="llama3-11b-instruct",  # 저장 디렉토리
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=5,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    push_to_hub=True,
    report_to="tensorboard",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field="messages",
    dataset_kwargs={"skip_prepare_dataset": True},
)
args.remove_unused_columns = False
