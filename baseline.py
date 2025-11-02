# baseline.py
from typing import Tuple
import time
import os
import wandb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from dataset import clean_text_keep_bang_qmark

def run_baseline(df, args):
    # --- split ---
    train_df, tmp_df = train_test_split(df, test_size=(1-args.train_frac),
                                        random_state=args.seed, stratify=df[args.label_col])
    val_size = args.val_frac/(args.val_frac+args.test_frac)
    val_df, test_df = train_test_split(tmp_df, test_size=(1-val_size),
                                       random_state=args.seed, stratify=tmp_df[args.label_col])

    print(f"Splits → train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

    # --- vectorizer ---
    vec = TfidfVectorizer(
        preprocessor=clean_text_keep_bang_qmark,
        ngram_range=(1,2),
        min_df=2, max_df=0.9, strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b|[!?]"  # keep ! and ? as standalone tokens
    )
    Xtr = vec.fit_transform(train_df[args.text_col])
    Xva = vec.transform(val_df[args.text_col])
    Xte = vec.transform(test_df[args.text_col])
    ytr = train_df[args.label_col].values
    yva = val_df[args.label_col].values
    yte = test_df[args.label_col].values

    # --- model ---
    if args.baseline_model == "lr":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None, solver="lbfgs", verbose=0)
    else:
        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=args.seed, n_jobs=-1, verbose=0)

    # --- W&B init (only if enabled) ---
    wandb_run = None
    if ("WANDB_DISABLED" not in os.environ) or (os.environ.get("WANDB_DISABLED","") != "true"):
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", None),
            entity=os.environ.get("WANDB_ENTITY", None),
            name=os.environ.get("WANDB_RUN_NAME", f"{args.baseline_model.upper()}-baseline"),
            config={
                "mode": "baseline",
                "baseline_model": args.baseline_model,
                "vectorizer": "tfidf(1,2)",
                "train_frac": args.train_frac,
                "val_frac": args.val_frac,
                "test_frac": args.test_frac,
                "seed": args.seed,
                "tfidf_vocab": len(vec.vocabulary_),
                **({"n_estimators": getattr(clf, "n_estimators", None)}),
            },
        )

    # --- fit ---
    clf.fit(Xtr, ytr)

    # --- eval & log ---
    for split_name, X, y in [("val", Xva, yva), ("test", Xte, yte)]:
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)
        p, r, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)

        print(f"[Baseline:{args.baseline_model}] {split_name.upper()} — Acc:{acc:.4f}  P:{p:.4f}  R:{r:.4f}  F1:{f1:.4f}")

        if wandb_run is not None:
            wandb.log({
                f"{split_name}/accuracy": acc,
                f"{split_name}/precision": p,
                f"{split_name}/recall": r,
                f"{split_name}/f1": f1,
            })
            if split_name == "test":
                wandb.log({
                    "test/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y,
                        preds=preds,
                        class_names=["not_clickbait", "clickbait"]
                    )
                })

    if wandb_run is not None:
        wandb.finish()
