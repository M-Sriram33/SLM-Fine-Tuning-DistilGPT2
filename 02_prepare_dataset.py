from datasets import load_dataset

# -----------------------------------
# 1. Load the dataset
# -----------------------------------

dataset = load_dataset("databricks/databricks-dolly-15k")["train"]

print("Total examples in dataset:", len(dataset))

# -----------------------------------
# 2. Shuffle and select a subset
# -----------------------------------

# We begin with 2,000 examples to keep training manageable
dataset = dataset.shuffle(seed=42).select(range(2000))

print("Examples selected for this project:", len(dataset))

# -----------------------------------
# 3. Split into training and validation
# -----------------------------------

split_dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42
)

train_dataset = split_dataset["train"]
validation_dataset = split_dataset["test"]

print("Training examples:", len(train_dataset))
print("Validation examples:", len(validation_dataset))

# -----------------------------------
# 4. Format each example
# -----------------------------------

def format_example(example):
    instruction = example["instruction"].strip()
    context = example["context"].strip()
    response = example["response"].strip()

    if context:
        formatted_text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Response:\n{response}"
        )
    else:
        formatted_text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}"
        )

    return {"text": formatted_text}


train_dataset = train_dataset.map(format_example)
validation_dataset = validation_dataset.map(format_example)

# -----------------------------------
# 5. Display an example
# -----------------------------------

print("\nFormatted training example:")
print(train_dataset[0]["text"])

# -----------------------------------
# 6. Save the processed datasets
# -----------------------------------

train_dataset.to_json("data/train.jsonl")
validation_dataset.to_json("data/validation.jsonl")

print("\nProcessed datasets saved successfully.")