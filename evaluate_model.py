import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to the trained checkpoint
MODEL_PATH = "outputs/distilgpt2-dolly/checkpoint-450"

# Select GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model from: {MODEL_PATH}")
print(f"Using device: {device}")

# Load tokenizer and fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

# GPT-2 does not have a default padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Test questions
test_questions = [
    "Summarize the importance of teamwork in the workplace.",
    "Write a short professional email requesting leave for two days.",
    "Explain why time management is important for students.",
    "Give five practical tips for improving communication skills.",
    "Rewrite this sentence professionally: I can't come to work today because I am sick.",
    "What are the advantages of learning a new language?",
    "Create a short motivational paragraph about overcoming failure.",
    "List three ways to reduce stress during exams.",
    "Explain the difference between a fact and an opinion.",
    "Write a short introduction for a presentation about technology."
]

for question in test_questions:
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Get only the tokens generated after the input prompt
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]

    # Decode only the newly generated tokens
    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print(f"Answer: {response}")