import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL = "distilgpt2"
FINE_TUNED_MODEL = "outputs/distilgpt2-dolly/checkpoint-450"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


def load_model(model_path):
    print(f"\nLoading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    return tokenizer, model


def generate_response(tokenizer, model, question):
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
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return response


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

base_tokenizer, base_model = load_model(BASE_MODEL)
fine_tokenizer, fine_model = load_model(FINE_TUNED_MODEL)

with open("comparison_results.txt", "w", encoding="utf-8") as file:
    for index, question in enumerate(test_questions, start=1):
        print("\n" + "=" * 70)
        print(f"Question {index}: {question}")

        base_response = generate_response(
            base_tokenizer,
            base_model,
            question
        )

        fine_response = generate_response(
            fine_tokenizer,
            fine_model,
            question
        )

        print("\n[BASE MODEL]")
        print(base_response)

        print("\n[FINE-TUNED MODEL]")
        print(fine_response)

        file.write("=" * 70 + "\n")
        file.write(f"Question {index}: {question}\n\n")
        file.write("[BASE MODEL]\n")
        file.write(base_response + "\n\n")
        file.write("[FINE-TUNED MODEL]\n")
        file.write(fine_response + "\n\n")

print("\nComparison completed.")
print("Results saved to comparison_results.txt")