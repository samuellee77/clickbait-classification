import argparse, os, re, random, wandb
import numpy as np, pandas as pd
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments, TrainerCallback)

# -----------------------
# Repro / cleaning
# -----------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# keep ! and ? ; remove digits & other punct; lowercase
_CLEAN_RE = re.compile(r"[^a-zA-Z!? \t]+")

def clean_text_keep_bang_qmark(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.lower()
    text = _CLEAN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

# -----------------------
# Datasets / metrics
# -----------------------
class HeadlineDS(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts, self.labels = list(texts), list(labels)
        self.tokenizer, self.max_len = tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], truncation=True, max_length=self.max_len, add_special_tokens=True)
        enc["labels"] = torch.tensor(int(self.labels[i]), dtype=torch.long)   # <-- was float
        return enc

def bin_metrics_from_logits(logits: np.ndarray, labels: np.ndarray, thr=0.5):
    probs = 1/(1+np.exp(-logits.reshape(-1)))
    preds = (probs >= thr).astype(int)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

class WandbEvalLogger(TrainerCallback):
    """Force-log eval metrics (esp. eval_loss) to W&B each epoch with nice names."""
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        # Ensure epoch is a defined step metric for nicer charts
        try:
            wandb.define_metric("epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("eval/*", step_metric="epoch")
        except Exception:
            pass

        payload = {"epoch": float(state.epoch) if state.epoch is not None else None}

        # Map HF eval_* keys -> val/* (and mirror to eval/*)
        key_map = {
            "eval_loss":      ["val/loss", "eval/loss"],
            "eval_accuracy":  ["val/accuracy", "eval/accuracy"],
            "eval_precision": ["val/precision", "eval/precision"],
            "eval_recall":    ["val/recall", "eval/recall"],
            "eval_f1":        ["val/f1", "eval/f1"],
        }
        for k, aliases in key_map.items():
            if k in metrics:
                for alias in aliases:
                    payload[alias] = float(metrics[k])

        # Log aligned to global_step so curves line up with train logs
        step = int(state.global_step) if state.global_step is not None else None
        wandb.log(payload, step=step)
# -----------------------
# Baselines
# -----------------------
def run_baseline(df, args):
    # --- split ---
    train_df, tmp_df = train_test_split(df, test_size=(1-args.train_frac),
                                        random_state=args.seed, stratify=df[args.label_col])
    val_size = args.val_frac/(args.val_frac+args.test_frac)
    val_df, test_df = train_test_split(tmp_df, test_size=(1-val_size),
                                       random_state=args.seed, stratify=tmp_df[args.label_col])

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
    if args.wandb_project and not args.wandb_disabled:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"{args.baseline_model.upper()}-baseline",
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

    if wandb_run is not None:
        wandb.finish()

# -----------------------
# Transformers
# -----------------------
def run_transformer(df, args):
    tok = AutoTokenizer.from_pretrained(args.pretrained, use_fast=True)
    if args.n_rows is not None:
        df = df[[args.text_col, args.label_col]].dropna()[:args.n_rows]
    else:
        df = df[[args.text_col, args.label_col]].dropna()
    df[args.text_col] = df[args.text_col].apply(clean_text_keep_bang_qmark)

    train_df, tmp_df = train_test_split(df, test_size=(1-args.train_frac),
                                        random_state=args.seed, stratify=df[args.label_col])
    val_size = args.val_frac/(args.val_frac+args.test_frac)
    val_df, test_df = train_test_split(tmp_df, test_size=(1-val_size),
                                       random_state=args.seed, stratify=tmp_df[args.label_col])

    ds_tr = HeadlineDS(train_df[args.text_col], train_df[args.label_col], tok, args.max_length)
    ds_va = HeadlineDS(val_df[args.text_col],   val_df[args.label_col],   tok, args.max_length)
    ds_te = HeadlineDS(test_df[args.text_col],  test_df[args.label_col],  tok, args.max_length)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained, num_labels=2
    )
    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8 if torch.cuda.is_available() else None)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

    report_to = "wandb" if args.wandb_project and not args.wandb_disabled else "none"
    if report_to == "wandb":
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity: os.environ["WANDB_ENTITY"] = args.wandb_entity
        os.environ["WANDB_RUN_NAME"] = args.wandb_run_name or f"{args.model_type}-{args.pretrained}"

    ta = TrainingArguments(
        output_dir=args.output_dir,
        run_name=(args.wandb_run_name or f"{args.model_type}-{args.pretrained}"),
        eval_strategy="epoch",     # or "steps" with eval_steps=200
        save_strategy="epoch",
        logging_strategy="steps",             # <-- prints non-zero train loss regularly
        disable_tqdm=False,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to=report_to,
        seed=args.seed,
    )
    
    trainer = Trainer(model=model, args=ta, train_dataset=ds_tr, eval_dataset=ds_va,
                      tokenizer=tok, data_collator=collator, compute_metrics=compute_metrics)
    if report_to == "wandb":
        trainer.add_callback(WandbEvalLogger())
    trainer.train()

    print("\n[Transformer] VAL:")
    val_metrics = trainer.evaluate(ds_va)
    print({k: round(float(v), 4) for k, v in val_metrics.items()
           if k.startswith(("eval_",)) or k in ("eval_accuracy","eval_precision","eval_recall","eval_f1")})

    print("\n[Transformer] TEST:")
    test_output = trainer.predict(ds_te)
    test_metrics = test_output.metrics
    print({k: round(float(v), 4) for k, v in test_metrics.items()
           if k.startswith(("test_", "eval_"))})

    # Optional: log test metrics + confusion matrix to W&B
    if report_to == "wandb":
        test_logits = test_output.predictions
        test_preds  = np.argmax(test_logits, axis=-1).astype(int)
        test_labels = test_output.label_ids.astype(int)

        wandb.log({ f"test/{k.replace('test_', '')}": float(v)
                    for k, v in test_metrics.items() if k.startswith("test_") })
        wandb.log({
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None, y_true=test_labels, preds=test_preds,
                class_names=["not_clickbait", "clickbait"]
            )
        })
# -----------------------
# Args / CSV loader
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser("Clickbait training pipeline (HF + baselines)")
    ap.add_argument("--data", type=str, required=True, help="CSV path with columns headline,clickbait")
    ap.add_argument("--text_col", type=str, default="headline")
    ap.add_argument("--label_col", type=str, default="clickbait")

    ap.add_argument("--n_rows", type=int, default=None, help="If set, limit to this many rows from CSV")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default="./outputs")

    ap.add_argument("--model_type", choices=["bert","distilbert","roberta","baseline"], required=True)
    ap.add_argument("--pretrained", type=str, default="bert-base-uncased")
    ap.add_argument("--max_length", type=int, default=128)

    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--lr_scheduler", choices=["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"],
                    default="linear")

    ap.add_argument("--baseline_model", choices=["lr","rf"], default="lr")
    # ADD after other args in parse_args()
    ap.add_argument("--wandb_project", type=str, default=None, help="W&B project name. If set, logs will be sent to Weights & Biases.")
    ap.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team/user). Optional.")
    ap.add_argument("--wandb_run_name", type=str, default=None, help="Optional run name.")
    ap.add_argument("--wandb_disabled", action="store_true", help="Disable W&B even if project is set.")
    
    args = ap.parse_args()

    # ADD after args = ap.parse_args()
    if args.wandb_project and not args.wandb_disabled:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity: os.environ["WANDB_ENTITY"] = args.wandb_entity
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # default pretrained per shortcut
    if args.model_type == "distilbert" and args.pretrained == "bert-base-uncased":
        args.pretrained = "distilbert-base-uncased"
    if args.model_type == "roberta" and args.pretrained == "bert-base-uncased":
        args.pretrained = "roberta-base"

    total = args.train_frac + args.val_frac + args.test_frac
    if not np.isclose(total, 1.0): raise ValueError("train/val/test fractions must sum to 1.0")
    return args

def load_csv_safely(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    # Handles embedded quotes/commas like in your sample
    df = pd.read_csv(
        path,
        engine="python",
        quotechar='"',
        doublequote=True,
        escapechar="\\",
        on_bad_lines="skip"
    )
    # basic checks / coercions
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Expected columns '{text_col}' and '{label_col}' in {path}. Found {df.columns.tolist()}")
    # force 0/1 ints if they came in as strings
    df[label_col] = df[label_col].astype(str).str.strip().replace({"0":"0","1":"1"}).astype(int)
    return df[[text_col, label_col]].dropna()

def main():
    set_seed()
    args = parse_args()
    set_seed(args.seed)

    df = load_csv_safely(args.data, args.text_col, args.label_col)

    if args.model_type == "baseline":
        run_baseline(df, args)
    else:
        run_transformer(df, args)

if __name__ == "__main__":
    main()
