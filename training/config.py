from transformers import TrainingArguments


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
