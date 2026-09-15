import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------
# 1. Load the base model
# -----------------------------------

model_name = "distilgpt2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPT-2 does not have a padding token by default
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

print("Model loaded successfully.")

# -----------------------------------
# 2. Select a test instruction
# -----------------------------------

prompt = """### Instruction:
Explain what machine learning is in simple terms.

### Response:
"""

# -----------------------------------
# 3. Tokenize the prompt
# -----------------------------------

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

# -----------------------------------
# 4. Generate a response
# -----------------------------------

print("\nGenerating response...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

# -----------------------------------
# 5. Decode and display response
# -----------------------------------

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nGenerated response:")
print(response)