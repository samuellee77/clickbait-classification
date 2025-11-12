# main.py
import os
import argparse
import numpy as np
import pandas as pd

from dataset import load_csv_safely
from baseline import run_baseline
from transformer import run_transformer

def optional_int(x: str):
    x = str(x).strip().lower()
    if x in {"none", "null", "nil", "" , "nan"}:
        return None
    v = int(x)
    return v

def parse_args():
    ap = argparse.ArgumentParser("Clickbait training pipeline (HF + baselines)")

    # Data
    ap.add_argument("--data", type=str, required=True, help="CSV path with columns headline,clickbait")
    ap.add_argument("--text_col", type=str, default="headline")
    ap.add_argument("--label_col", type=str, default="clickbait")

    # Splits
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)

    # General
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default="./outputs")
    ap.add_argument("--n_rows", type=int, default=None, help="Optional: use only the first N rows for quick experiments")

    # Mode
    ap.add_argument("--model_type", choices=["bert","distilbert","roberta","baseline"], required=True)
    ap.add_argument("--pretrained", type=str, default="bert-base-uncased")
    ap.add_argument("--max_length", type=int, default=128)

    # Optim
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--lr_scheduler", choices=["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"],
                    default="linear")

    # Baseline options
    ap.add_argument("--baseline_model", choices=["lr","rf"], default="lr")
    ap.add_argument("--tfidf_min_df", type=int, default=2)
    ap.add_argument("--tfidf_max_df", type=float, default=0.9)
    ap.add_argument("--tfidf_ngram_max", type=int, default=2)  # ngram_range=(1, tfidf_ngram_max)
    ap.add_argument("--tfidf_sublinear_tf", action="store_true")
    ap.add_argument("--tfidf_use_idf", action="store_true")

    ap.add_argument("--lr_C", type=float, default=1.0)
    ap.add_argument("--lr_penalty", choices=["l2"], default="l2")
    ap.add_argument("--lr_max_iter", type=int, default=1000)

    ap.add_argument("--rf_estimators", type=int, default=400)
    ap.add_argument("--rf_max_depth", type=optional_int, default=None)
    ap.add_argument("--rf_min_samples_split", type=int, default=2)
    ap.add_argument("--rf_min_samples_leaf", type=int, default=1)
    ap.add_argument("--rf_max_features", type=str, default="sqrt")

    # W&B
    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_disabled", action="store_true")

    args = ap.parse_args()

    # default pretrained per shortcut
    if args.model_type == "distilbert" and args.pretrained == "bert-base-uncased":
        args.pretrained = "distilbert-base-uncased"
    if args.model_type == "roberta" and args.pretrained == "bert-base-uncased":
        args.pretrained = "roberta-base"

    total = args.train_frac + args.val_frac + args.test_frac
    if not np.isclose(total, 1.0):
        raise ValueError("train/val/test fractions must sum to 1.0")

    # Wire W&B env if enabled
    if args.wandb_project and not args.wandb_disabled:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity: os.environ["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_run_name: os.environ["WANDB_RUN_NAME"] = args.wandb_run_name
    else:
        os.environ["WANDB_DISABLED"] = "true"

    return args

def main():
    args = parse_args()
    df = load_csv_safely(args.data, args.text_col, args.label_col)
    if args.n_rows:
        df = df.head(args.n_rows).copy()

    if args.model_type == "baseline":
        run_baseline(df, args)
    else:
        run_transformer(df, args)

if __name__ == "__main__":
    main()
