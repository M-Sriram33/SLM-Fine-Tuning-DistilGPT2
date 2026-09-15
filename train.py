import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

# ============================================================
# 1. Configuration
# ============================================================

MODEL_NAME = "distilgpt2"
OUTPUT_DIR = "./outputs/distilgpt2-dolly"

# ============================================================
# 2. Load the prepared datasets
# ============================================================

print("Loading training dataset...")

train_dataset = load_dataset(
    "json",
    data_files="data/train.jsonl"
)["train"]

print("Loading validation dataset...")

validation_dataset = load_dataset(
    "json",
    data_files="data/validation.jsonl"
)["train"]

print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(validation_dataset)}")

# ============================================================
# 3. Load tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPT-2 models do not have a padding token by default.
# We use the EOS token as the padding token.
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 4. Load model
# ============================================================

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.config.pad_token_id = tokenizer.pad_token_id

# ============================================================
# 5. Tokenize datasets
# ============================================================

print("\nTokenizing datasets...")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512
    )


tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

tokenized_validation = validation_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=validation_dataset.column_names
)

print("Tokenization completed.")

# ============================================================
# 6. Create data collator
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# mlm=False means this is causal language-model training.
# The model learns to predict the next token.

# ============================================================
# 7. Training arguments
# ============================================================

print("\nConfiguring training arguments...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Number of complete passes through the dataset
    num_train_epochs=2,

    # Number of examples processed at once
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,

    # Simulates a larger batch size
    gradient_accumulation_steps=4,

    # Learning rate
    learning_rate=5e-5,

    # Helps reduce overfitting
    weight_decay=0.01,

    # Display training information every 25 steps
    logging_steps=25,

    # Evaluate after each epoch
    eval_strategy="epoch",

    # Save a checkpoint after each epoch
    save_strategy="epoch",

    # Keep only the latest two checkpoints
    save_total_limit=2,

    # Load the best checkpoint at the end
    load_best_model_at_end=True,

    # Disable external logging services
    report_to="none",

    # Use mixed precision only when a CUDA GPU is available
    fp16=torch.cuda.is_available()
)

# ============================================================
# 8. Create Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    processing_class=tokenizer,
    data_collator=data_collator
)

# ============================================================
# 9. Start training
# ============================================================

print("\n" + "=" * 60)
print("STARTING FINE-TUNING")
print("=" * 60)

trainer.train()

# ============================================================
# 10. Save the final model
# ============================================================

print("\nSaving fine-tuned model...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 60)
print("FINE-TUNING COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Model saved at: {OUTPUT_DIR}")