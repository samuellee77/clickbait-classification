# Clickbait Classification

This project fine-tunes transformer models and trains classical ML baselines to classify whether a news headline is **clickbait** or not.

---

## Project Structure

```txt
clickbait-classification/
│
├── main.py             # entry point with argument parser
├── dataset.py          # text cleaning and dataset loading utilities
├── baseline.py         # classical baselines (TF-IDF + Logistic Regression / Random Forest)
├── transformer.py      # transformer training (BERT / DistilBERT / RoBERTa)
├── data.csv            # your dataset (headline, clickbait)
└── README.md           # you are here
```

---

## Data Format

The dataset must be a **CSV** file with at least two columns:

| headline                                               | clickbait |
| ------------------------------------------------------ | --------- |
| `"Jurassic World" uses bad science - Business Insider` | 1         |
| `"Apprentice" contestant sues Trump for defamation`    | 0         |

* **headline** → string title
* **clickbait** → `1` for clickbait, `0` otherwise

---

## Preprocessing

Implemented in **dataset.py**

* Lowercase conversion
* Removes digits and most punctuation
* Keeps `!` and `?` because they are strong clickbait cues
* Cleans repeated whitespace

---

## Models

### Transformer Models (in `transformer.py`)

Fine-tune pretrained models from the Hugging Face library:

* `bert-base-uncased`
* `distilbert-base-uncased`
* `roberta-base`

Uses binary classification head (`num_labels=2`) trained with:

* **Loss:** Binary cross-entropy
* **Optimizer:** AdamW
* **Scheduler:** configurable (linear, cosine, etc.)
* **Evaluation:** Accuracy, Precision, Recall, F1

All training progress and metrics can be logged to **Weights & Biases (W&B)**.

### Baselines (in `baseline.py`)

Classical models trained on TF-IDF features:

* **Logistic Regression** (`--baseline_model lr`)
* **Random Forest** (`--baseline_model rf`)

Evaluated on accuracy, precision, recall, and F1.

---

## Running Experiments

### Transformers

#### Example: DistilBERT (fine-tuning)

```bash
python main.py \
  --data data.csv \
  --model_type distilbert \
  --epochs 3 \
  --lr 3e-5 \
  --batch_size 32 \
  --wandb_project clickbait-exp \
  --wandb_run_name "DistilBERT" \
  --n_rows 1000
```

#### Example: RoBERTa

```bash
python main.py \
  --data data.csv \
  --model_type roberta \
  --epochs 4 \
  --lr 2e-5 \
  --batch_size 16 \
  --wandb_project clickbait-exp \
  --wandb_run_name "RoBERTa"
```

---

### Baseline Models

#### Logistic Regression

```bash
python main.py \
  --data data.csv \
  --model_type baseline \
  --baseline_model lr \
  --wandb_project clickbait-exp \
  --wandb_run_name "LR-baseline"
```

#### Random Forest

```bash
python main.py \
  --data data.csv \
  --model_type baseline \
  --baseline_model rf \
  --wandb_project clickbait-exp \
  --wandb_run_name "RF-baseline"
```

---

## Arguments Overview

| Argument           | Description                                             | Default                     |
| ------------------ | ------------------------------------------------------- | --------------------------- |
| `--data`           | Path to dataset CSV                                     | **Required**                |
| `--text_col`       | Name of text column                                     | `headline`                  |
| `--label_col`      | Name of label column                                    | `clickbait`                 |
| `--train_frac`     | Fraction for training set                               | `0.8`                       |
| `--val_frac`       | Fraction for validation set                             | `0.1`                       |
| `--test_frac`      | Fraction for test set                                   | `0.1`                       |
| `--seed`           | Random seed                                             | `42`                        |
| `--output_dir`     | Output directory for checkpoints                        | `./outputs`                 |
| `--n_rows`         | Use only first *N* rows for debugging                   | `None`                      |
| `--model_type`     | `bert`, `distilbert`, `roberta`, `baseline`             | **Required**                |
| `--pretrained`     | Pretrained checkpoint name                              | auto-selected by model type |
| `--batch_size`     | Batch size                                              | `16`                        |
| `--epochs`         | Number of epochs                                        | `3`                         |
| `--lr`             | Learning rate                                           | `2e-5`                      |
| `--warmup_ratio`   | Warmup ratio for scheduler                              | `0.06`                      |
| `--lr_scheduler`   | Scheduler type (`linear`, `cosine`, `polynomial`, etc.) | `linear`                    |
| `--baseline_model` | `lr` (LogReg) or `rf` (RandomForest)                    | `lr`                        |
| `--wandb_project`  | W&B project name (optional)                             | `None`                      |
| `--wandb_entity`   | W&B entity / team name                                  | `None`                      |
| `--wandb_run_name` | W&B run name                                            | `None`                      |
| `--wandb_disabled` | Disable W&B logging                                     | `False`                     |

---

## Weights & Biases Integration

* Runs automatically call `wandb.init()` if a project name is given.
* Logs include:

  * **Train:** loss, accuracy, precision, recall, F1
  * **Eval:** loss, accuracy, precision, recall, F1
  * **Test:** loss, accuracy, precision, recall, F1
* Confusion matrices and PR curves are logged for test results.

> If you don’t want to use W&B, add `--wandb_disabled`.

---

## Development Notes

* Uses `AutoTokenizer` and `AutoModelForSequenceClassification`.
* Custom callback logs per-epoch train + validation metrics.
* TF-IDF baselines use bigrams `(1,2)` and balanced class weights.
* Works with small subsets via `--n_rows` for quick tests.

---

## Example Output

```txt
Splits → train:800  val:100  test:100
{'epoch': 1.0, 'train/loss': 0.34, 'train/accuracy': 0.87, 'train/f1': 0.88}
{'epoch': 1.0, 'val/loss': 0.43, 'val/accuracy': 0.78, 'val/f1': 0.78}
...
[Transformer] TEST:
{'test_loss': 0.31, 'test_accuracy': 0.89, 'test_f1': 0.90}
```
