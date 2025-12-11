# Detecting Clickbait with RoBERTa: Improving Classification over Traditional Baselines

This repository provides a reproducible, modular pipeline for training and evaluating clickbait-classification models using either transformer architectures (BERT, DistilBERT, RoBERTa) or traditional TF-IDF baselines (Logistic Regression, Random Forest). The codebase includes consistent preprocessing, safe dataset loading, Weights & Biases (W&B) integration, configurable hyperparameters, and sweep configurations for automated experimentation.

---

## Repository Structure

```txt
clickbait-classification/
│
├── main.py                # Main entry point; routes to model type
├── dataset.py             # Safe CSV loader and text normalization
├── baseline.py            # TF-IDF + Logistic Regression / Random Forest
├── transformer.py         # Transformer fine-tuning pipeline
├── environment.yml        # Reproducible conda environment
│
├── sweep.yaml             # Sweep for transformer models
├── sweep_lr.yaml          # Sweep for Logistic Regression
├── sweep_rf.yaml          # Sweep for Random Forest
│
├── data.csv               # Example dataset (headline, clickbait)
└── README.md              # Documentation (this file)
```

---

## Installation and Environment Setup

### Clone the repository

```bash
git clone https://github.com/samuellee77/clickbait-classification.git
cd clickbait-classification
```

### Create and activate environment

```bash
conda env create -f environment.yml
conda activate clickbait-nlp
```

---

## Text Preprocessing

All text is preprocessed prior to model ingestion:

* conversion to lowercase
* removal of digits and non-alphabetic characters
* preservation of `!` and `?` as meaningful features
* normalization of whitespace
* robust handling of null and malformed entries

The same cleaning function is used for both baseline and transformer pipelines to ensure consistency.

---

## Models

### Transformer Models

Located in `transformer.py`. Supports:

* BERT (`bert-base-uncased`)
* DistilBERT (`distilbert-base-uncased`)
* RoBERTa (`roberta-base`)

Features:

* Hugging Face `Trainer` with custom metrics
* automatic padding (`DataCollatorWithPadding`)
* evaluation on validation and test splits
* mixed precision (`fp16`) when CUDA is available
* early selection of best model based on F1
* full W&B integration

Automatic model selection:

| `--model_type` | Default `--pretrained`    |
| -------------- | ------------------------- |
| `bert`         | `bert-base-uncased`       |
| `distilbert`   | `distilbert-base-uncased` |
| `roberta`      | `roberta-base`            |

---

### Baseline Models

Located in `baseline.py`.

Supported classifiers:

* Logistic Regression
* Random Forest

TF-IDF vectorizer configuration:

* adjustable n-gram range (`1` to `tfidf_ngram_max`)
* configurable `min_df`, `max_df`
* optional IDF usage and sublinear TF scaling
* custom tokenization preserving `!` and `?`

---

## Running Experiments

### Transformer Training Example

```bash
python main.py \
  --data data.csv \
  --model_type roberta \
  --epochs 4 \
  --lr 2e-5 \
  --batch_size 16 \
  --max_length 128 \
  --wandb_project clickbait-exp
```

### Baseline Training Example (Logistic Regression)

```bash
python main.py \
  --data data.csv \
  --model_type baseline \
  --baseline_model lr \
  --wandb_project clickbait-exp
```

### Baseline Training Example (Random Forest)

```bash
python main.py \
  --data data.csv \
  --model_type baseline \
  --baseline_model rf \
  --wandb_project clickbait-exp
```

---

## Command-Line Arguments

### Core Arguments

| Argument                                      | Description                         | Default         |
| --------------------------------------------- | ----------------------------------- | --------------- |
| `--data`                                      | Path to CSV dataset                 | required        |
| `--text_col`                                  | Name of text column                 | headline        |
| `--label_col`                                 | Name of label column                | clickbait       |
| `--train_frac` / `--val_frac` / `--test_frac` | Data split fractions; must sum to 1 | 0.8 / 0.1 / 0.1 |
| `--n_rows`                                    | Use first N rows for debugging      | None            |
| `--output_dir`                                | Directory for model checkpoints     | ./outputs       |

### Transformer-Specific Arguments

| Argument         | Purpose                   | Default       |
| ---------------- | ------------------------- | ------------- |
| `--model_type`   | bert, distilbert, roberta | required      |
| `--pretrained`   | HF checkpoint             | auto-selected |
| `--max_length`   | Token sequence length     | 128           |
| `--lr`           | Learning rate             | 2e-5          |
| `--epochs`       | Training epochs           | 3             |
| `--batch_size`   | Per-device batch size     | 16            |
| `--lr_scheduler` | Scheduler type            | linear        |
| `--warmup_ratio` | Warmup proportion         | 0.06          |

### Baseline-Specific Arguments

TF-IDF:

* `tfidf_min_df`
* `tfidf_max_df`
* `tfidf_ngram_max`
* `tfidf_use_idf`
* `tfidf_sublinear_tf`

Logistic Regression:

* `lr_C`
* `lr_penalty`
* `lr_max_iter`

Random Forest:

* `rf_estimators`
* `rf_max_depth`
* `rf_min_samples_split`
* `rf_min_samples_leaf`
* `rf_max_features`

### Weights & Biases Arguments

* `--wandb_project`
* `--wandb_entity`
* `--wandb_run_name`
* `--wandb_disabled` (forces `WANDB_DISABLED="true"`)

---

## Hyperparameter Sweeps

### Transformer Sweep (`sweep.yaml`)

Explores:

* learning rate
* batch size
* warmup ratio
* maximum sequence length
* number of rows (subset training)

### Logistic Regression Sweep (`sweep_lr.yaml`)

Explores:

* TF-IDF configurations
* regularization strength
* maximum iterations

### Random Forest Sweep (`sweep_rf.yaml`)

Explores:

* n_estimators
* max_depth
* min_samples_split

All sweeps optimize for validation F1.
