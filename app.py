import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

with st.sidebar:
    st.header("About the Project")

    st.write("**Model:** DistilGPT-2")
    st.write("**Dataset:** Databricks Dolly")
    st.write("**Training Samples:** Approximately 2,000")
    st.write("**Device:** CPU/GPU automatically selected")

    st.markdown("---")

    st.subheader("Example Prompts")
    st.write("- Summarize the importance of teamwork in the workplace.")
    st.write("- Give three interview preparation tips")
    st.write("- Explain supervised learning")
    st.write("- Write a short Python function")

    st.markdown("---")

    st.warning(
        "This model is experimental and may produce incomplete, "
        "repetitive, or inaccurate responses."
    )
# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = "outputs/distilgpt2-dolly/checkpoint-450"

st.set_page_config(
    page_title="Fine-Tuned DistilGPT-2",
    page_icon="🤖",
    layout="centered"
)


# -----------------------------
# Load Model and Tokenizer
# -----------------------------

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use tokenizer from checkpoint if available.
    # Otherwise, use the original DistilGPT-2 tokenizer.
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    # DistilGPT-2 does not have a default padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    return tokenizer, model, device


# -----------------------------
# Generate Response
# -----------------------------

def generate_response(
    question,
    tokenizer,
    model,
    device,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=False
):
    prompt = (
        "### Instruction:\n"
        f"{question}\n\n"
        "### Response:\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only newly generated tokens
    generated_tokens = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    # Remove any accidental next sections
    if "### Instruction:" in response:
        response = response.split("### Instruction:")[0].strip()

    if "### Response:" in response:
        response = response.split("### Response:")[0].strip()

    return response


# -----------------------------
# Streamlit Interface
# -----------------------------

st.title("🤖 Fine-Tuned DistilGPT-2")
st.write(
    "A small language model fine-tuned on the Databricks Dolly instruction dataset."
)



with st.spinner("Loading model..."):
    tokenizer, model, device = load_model()

st.success(f"Model loaded successfully on: {device}")

st.subheader("Ask the Model")

question = st.text_area(
    "Enter your instruction or question:",
    placeholder="Explain what machine learning is in simple terms.",
    height=120
)

col1, col2 = st.columns(2)

with col1:
    max_new_tokens = st.slider(
        "Maximum response length",
        min_value=32,
        max_value=200,
        value=100,
        step=8
    )

with col2:
    use_sampling = st.checkbox(
        "Enable sampling",
        value=False
    )

temperature = 0.7

if use_sampling:
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.1
    )

if st.button("Generate Response", type="primary"):
    if not question.strip():
        st.warning("Please enter a question or instruction.")
    else:
        with st.spinner("Generating response..."):
            response = generate_response(
                question=question,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=use_sampling
            )

        st.subheader("Model Response")

        if response:
            st.write(response)
        else:
            st.warning("The model did not generate a response.")