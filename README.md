# Clickbait Classification

This repository fine-tunes transformer models (BERT, DistilBERT, RoBERTa) and trains traditional ML baselines (Logistic Regression, Random Forest) to classify whether a news headline is *clickbait* or not.

## Data Structure

The dataset is from <https://www.kaggle.com/datasets/vikassingh1996/news-clickbait-dataset> The dataset is structured with two columns:

| headline | clickbait |
|-----------|------------|
| `"Jurassic World" uses bad science - Business Insider` | 1 |
| `"Bring it on": Students sue Trump administration over climate change` | 0 |

---

## Environment Setup

### Prerequistes

- Git
- Miniconda
- Weight & Biases acoount

### Installation

```shell
git clone https://github.com/samuellee77/clickbait-classification.git
cd clickbait-classification
```

Create the environment:

```shell
conda env create -f environment.yml
conda activate clickbait-nlp
```

---

## Running Experiments

All experiments are launched with the same script:

```bash
python train_clickbait.py --data path/to/data.csv --model_type <type> [options]
```

### Supported Model Types

| Model Type   | Description                                   |
| ------------ | --------------------------------------------- |
| `baseline`   | TF-IDF + Logistic Regression or Random Forest |
| `bert`       | Fine-tune BERT-base-uncased                   |
| `distilbert` | Fine-tune DistilBERT-base-uncased             |
| `roberta`    | Fine-tune RoBERTa-base                        |

---

## Example Commands

### 1. Baseline: Logistic Regression

```bash
python train_clickbait.py --data data.csv --model_type baseline --baseline_model lr
```

### 2. Baseline: Random Forest

```bash
python train_clickbait.py --data data.csv --model_type baseline --baseline_model rf
```

### 3. Transformer: BERT-base

```bash
python train_clickbait.py --data data.csv --model_type bert --epochs 3 \
  --wandb_project <your project name> --wandb_run_name "BERT-base"
```

### 4. Transformer: DistilBERT

```bash
python train_clickbait.py --data data.csv --model_type distilbert \
  --epochs 3 --lr 3e-5 --batch_size 32 \
  --wandb_project <your project name> --wandb_run_name "DistilBERT"
```

### 5. Transformer: RoBERTa-base

```bash
python train_clickbait.py --data data.csv --model_type roberta \
  --epochs 4 --lr 2e-5 --batch_size 16 \
  --wandb_project <your project name> --wandb_run_name "RoBERTa-base"
```

---

## W&B (Weights & Biases) Logging

You can track metrics (train/val loss, accuracy, precision, recall, F1, etc.) on [wandb.ai](https://wandb.ai/).

To enable logging:

```bash
python train_clickbait.py --data data.csv --model_type bert \
  --wandb_project clickbait-exp --wandb_entity your_username \
  --wandb_run_name "bert-run-1"
```

If you don’t want to log to W&B:

```bash
--wandb_disabled
```

W&B logs include:

- `train/loss`, `val/loss`
- `val/accuracy`, `val/precision`, `val/recall`, `val/f1`
- `test/accuracy`, `test/confusion_matrix`

---

## Arguments Overview

| Argument           | Type  | Default             | Description                                        |
| ------------------ | ----- | ------------------- | -------------------------------------------------- |
| `--data`           | str   | **Required**        | Path to dataset CSV file                           |
| `--text_col`       | str   | `headline`          | Column name for text input                         |
| `--label_col`      | str   | `clickbait`         | Column name for label                              |
| `--model_type`     | str   | **Required**        | One of `baseline`, `bert`, `distilbert`, `roberta` |
| `--pretrained`     | str   | `bert-base-uncased` | Hugging Face model checkpoint                      |
| `--baseline_model` | str   | `lr`                | For baseline runs: `lr` or `rf`                    |
| `--epochs`         | int   | `3`                 | Number of training epochs                          |
| `--batch_size`     | int   | `16`                | Batch size per device                              |
| `--lr`             | float | `2e-5`              | Learning rate                                      |
| `--max_length`     | int   | `128`               | Max token length per headline                      |
| `--train_frac`     | float | `0.8`               | Fraction of training data                          |
| `--val_frac`       | float | `0.1`               | Fraction of validation data                        |
| `--test_frac`      | float | `0.1`               | Fraction of test data                              |
| `--output_dir`     | str   | `./outputs`         | Directory for model checkpoints and logs           |
| `--seed`           | int   | `42`                | Random seed for reproducibility                    |
| `--warmup_ratio`   | float | `0.06`              | Warmup fraction for learning rate scheduler        |
| `--lr_scheduler`   | str   | `linear`            | Scheduler type (`linear`, `cosine`, etc.)          |
| `--wandb_project`  | str   | None                | W&B project name                                   |
| `--wandb_entity`   | str   | None                | W&B username or team                               |
| `--wandb_run_name` | str   | None                | Custom run name on W&B                             |
| `--wandb_disabled` | flag  | False               | Disable W&B logging                                |

---

## Outputs

After each run, you’ll get:

- **Console summary**:

  ```shell
  [Transformer] VAL: {'eval_loss': 0.524, 'eval_accuracy': 0.873, 'eval_f1': 0.861}
  [Transformer] TEST: {'eval_loss': 0.511, 'eval_accuracy': 0.880, 'eval_f1': 0.870}
  ```

- **W&B dashboard** with training curves and confusion matrices
