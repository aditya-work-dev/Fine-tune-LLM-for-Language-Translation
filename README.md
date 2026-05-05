# Fine-Tuning an LLM for German-to-French Translation

This project fine-tunes a large language model for **German → French machine translation** using parameter-efficient fine-tuning. The workflow uses a quantized `Qwen/Qwen2.5-3B` model, LoRA adapters, Hugging Face datasets, and BLEU evaluation to compare different training-data strategies.

The project was built as an end-to-end notebook covering dataset preparation, baseline evaluation, supervised fine-tuning, synthetic data generation, merged-dataset training, and translation quality evaluation.

---

## Project Overview

The goal of this project is to adapt a general-purpose LLM for a specific translation task: translating German sentences into French.

Instead of fully fine-tuning all model parameters, the project uses **QLoRA-style fine-tuning** with 4-bit quantization and LoRA adapters. This makes the training process more memory-efficient and practical to run in a notebook/Colab GPU environment.

---

## Key Features

- Loads the `Qwen/Qwen2.5-3B` causal language model.
- Uses 4-bit quantization with `bitsandbytes`.
- Applies LoRA fine-tuning using `peft`.
- Uses `trl.SFTTrainer` for supervised fine-tuning.
- Prepares German-French translation data in instruction format.
- Evaluates model outputs using BLEU score.
- Generates additional synthetic translation data using the Groq API.
- Compares multiple model/data configurations.
- Saves fine-tuned LoRA adapters and tokenizer files.

---

## Tech Stack

- **Language:** Python
- **Notebook Environment:** Google Colab / Jupyter Notebook
- **Model:** `Qwen/Qwen2.5-3B`
- **Libraries:**
  - `torch`
  - `transformers`
  - `datasets`
  - `bitsandbytes`
  - `peft`
  - `trl`
  - `evaluate`
  - `pandas`
  - `matplotlib`
  - `groq`

---

## Dataset

The project uses German-French sentence pairs. Each sample contains a German sentence and its French translation.

Example format:

```json
{
  "id": "108376",
  "translation": {
    "de": "Wir wohnen in Afrika.",
    "fr": "Nous habitons en Afrique."
  }
}
```

### Dataset Files

| File | Description | Size |
|---|---|---:|
| `First_training_set.json` | Initial German-French training set sampled from the Tatoeba dataset | 800 samples |
| `First_testing_set.json` | Test set used for BLEU evaluation | 200 samples |
| `Synthetic_Data.json` | Synthetic German-French translation data generated for domain expansion | 1,600 samples |
| `Merged_Dataset_Model_D.json` | Combined dataset containing original + synthetic data | 2,400 samples |

The synthetic dataset focuses mainly on travel, food, culture, shopping, entertainment, and tourism-related sentences.

---

## Methodology

### 1. Baseline Model Evaluation

The base `Qwen/Qwen2.5-3B` model is first evaluated on the German-French test set without fine-tuning. This gives a baseline BLEU score.

### 2. Instruction Formatting

Each sentence pair is converted into an instruction-style prompt:

```text
Translate German sentences to French.

German: <German sentence>
French: <French translation>
```

This allows the causal language model to learn the translation task through supervised fine-tuning.

### 3. Quantization

The model is loaded in 4-bit precision using `BitsAndBytesConfig`:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

This reduces GPU memory usage and makes fine-tuning more practical.

### 4. LoRA Fine-Tuning

LoRA adapters are applied to the attention projection layers:

```python
LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 5. Supervised Fine-Tuning

The model is trained using `SFTTrainer` with the following main settings:

| Parameter | Value |
|---|---:|
| Epochs | 2 |
| Batch size | 8 |
| Learning rate | `1e-4` |
| Max sequence length | 1024 |
| Optimizer | `paged_adamw_32bit` |
| Precision | FP16 |
| Seed | 42 |

### 6. Evaluation

The model is evaluated using BLEU score on the held-out test set. BLEU compares generated French translations with reference French translations.

---

## Results

| Experiment | Training Data | BLEU Score | BLEU % |
|---|---|---:|---:|
| Model A - Baseline | No fine-tuning | 0.3748 | 37.48% |
| Model B - Fine-tuned | 800 original sentence pairs | 0.4220 | 42.20% |
| Model C - Synthetic fine-tuned | 1,600 synthetic sentence pairs | 0.4660 | 46.60% |
| Model D - Merged dataset | 2,400 original + synthetic pairs | 0.3748 | 37.48% |

The best-performing setup in this experiment was **Model C**, trained on the synthetic dataset, with a BLEU score of **46.60%**.

---

## Example Output

Input:

```text
German: Zweifel zerfrisst wie ein Holzwurm auch den stärksten Balken.
```

Generated translation:

```text
Les doutes rongent comme un ver de bois même le plus solide des piliers.
```

---

## Repository Structure

Recommended structure for this repository:

```text
.
├── Fine_tune_LLM_for_Language_Translation.ipynb
├── data/
│   ├── First_training_set.json
│   ├── First_testing_set.json
│   ├── Synthetic_Data.json
│   └── Merged_Dataset_Model_D.json
├── README.md
└── .gitignore
```

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/aditya-work-dev/<repo-name>.git
cd <repo-name>
```

### 2. Install Dependencies

```bash
pip install torch transformers datasets bitsandbytes peft trl evaluate pandas matplotlib groq accelerate
```

### 3. Add API Keys Safely

This project uses Hugging Face and Groq credentials inside the notebook.

Do **not** hard-code API keys in the notebook before pushing to GitHub.

Use environment variables or Colab secrets instead:

```python
import os

HF_API_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

### 4. Run the Notebook

Open the notebook:

```bash
jupyter notebook Fine_tune_LLM_for_Language_Translation.ipynb
```

Or run it in Google Colab with a GPU runtime.

---


## What I Learned

Through this project, I practiced:

- Fine-tuning LLMs for a specific NLP task.
- Using QLoRA/LoRA for memory-efficient training.
- Working with Hugging Face `transformers`, `datasets`, `peft`, and `trl`.
- Preparing translation datasets in instruction format.
- Evaluating machine translation using BLEU score.
- Comparing model performance across different dataset strategies.
- Understanding how synthetic data can affect model performance.

---

## Limitations

- The training and test datasets are relatively small.
- BLEU score alone does not fully capture translation quality.
- Some synthetic translations may contain grammatical or semantic errors.
- The merged dataset did not improve performance in this experiment, which suggests that dataset quality and consistency matter more than simply increasing dataset size.
- The project is currently notebook-based and does not include a deployed translation API or web interface.

---

## Future Improvements

- Clean and validate synthetic data before training.
- Add more evaluation metrics such as chrF, METEOR, or COMET.
- Compare with dedicated translation models such as MarianMT or mBART.
- Increase dataset size and improve data diversity.
- Add a simple Gradio or Streamlit interface for live translation.
- Package the training pipeline into reusable Python scripts.
- Upload the LoRA adapter to Hugging Face Hub.

---

## Author

**Aditya Utpat**

GitHub: [aditya-work-dev](https://github.com/aditya-work-dev)
