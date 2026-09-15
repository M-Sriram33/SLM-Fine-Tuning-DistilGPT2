# Fine-Tuning DistilGPT-2 on Instruction Data

This project demonstrates the fine-tuning of the pretrained **DistilGPT-2** language model on the **Databricks Dolly instruction-following dataset**.

The project includes dataset preparation, model fine-tuning, evaluation, baseline comparison, and an interactive Streamlit application for text generation.

---

## Features

- Load and prepare instruction-response data
- Fine-tune DistilGPT-2 using Hugging Face Transformers
- Evaluate the fine-tuned model
- Compare baseline and fine-tuned model outputs
- Generate responses through a Streamlit interface

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Streamlit
- Pandas
- NumPy

---

## Project Structure

```text
SLM Fine Tuning/
│
├── data/
│   ├── train.jsonl
│   └── validation.jsonl
│
├── 01_load_dataset.py
├── 02_prepare_dataset.py
├── 03_test_model.py
├── train.py
├── evaluate_model.py
├── compare_models.py
├── app.py
│
├── baseline_results.txt
├── comparison_results.txt
├── evaluation_summary.txt
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

This project uses the **Databricks Dolly instruction-following dataset**.

The dataset contains instruction-response examples used to fine-tune the model for instruction-based text generation.

The prepared dataset is stored in JSONL format:

```text
data/train.jsonl
data/validation.jsonl
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Load the Dataset

```bash
python 01_load_dataset.py
```

### 2. Prepare the Dataset

```bash
python 02_prepare_dataset.py
```

### 3. Test the Pretrained Model

```bash
python 03_test_model.py
```

### 4. Fine-Tune the Model

```bash
python train.py
```

### 5. Evaluate the Model

```bash
python evaluate_model.py
```

### 6. Compare Model Results

```bash
python compare_models.py
```

### 7. Launch the Streamlit Application

```bash
streamlit run app.py
```

---

## Results

The project stores evaluation and comparison results in the following files:

```text
baseline_results.txt
comparison_results.txt
evaluation_summary.txt
```

The fine-tuned model is evaluated based on validation performance and generated response quality.

---

## Limitations

- DistilGPT-2 has limited model capacity compared with larger language models.
- The model may generate repetitive or factually incorrect responses.
- Fine-tuning performance depends on dataset size and training configuration.
- Trained model checkpoints are excluded from GitHub because of their large file size.

---

## Future Improvements

- Fine-tune on a larger dataset
- Experiment with LoRA or QLoRA
- Improve response evaluation
- Add conversation history
- Deploy the Streamlit application online
- Apply quantization for efficient inference

---

## Author

**M.Sriram**

