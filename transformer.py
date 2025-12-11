# transformer.py
import os
import wandb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)

from dataset import clean_text_keep_bang_qmark

# ---- Dataset class (transformer) ----
class HeadlineDS(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts, self.labels = list(texts), list(labels)
        self.tokenizer, self.max_len = tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], truncation=True, max_length=self.max_len, add_special_tokens=True)
        enc["labels"] = torch.tensor(int(self.labels[i]), dtype=torch.long)  # CrossEntropy labels
        return enc

# ---- Compute metrics ----
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# ---- Main entry for transformer training ----
def run_transformer(df, args):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Using device:", device)
    tok = AutoTokenizer.from_pretrained(args.pretrained, use_fast=True)

    # Clean
    df = df[[args.text_col, args.label_col]].dropna()
    df[args.text_col] = df[args.text_col].apply(clean_text_keep_bang_qmark)

    # Split
    train_df, tmp_df = train_test_split(df, test_size=(1-args.train_frac),
                                        random_state=args.seed, stratify=df[args.label_col])
    val_size = args.val_frac/(args.val_frac+args.test_frac)
    val_df, test_df = train_test_split(tmp_df, test_size=(1-val_size),
                                       random_state=args.seed, stratify=tmp_df[args.label_col])
    print(f"Splits → train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

    # Datasets
    train_ds = HeadlineDS(train_df[args.text_col], train_df[args.label_col], tok, args.max_length)
    val_ds   = HeadlineDS(val_df[args.text_col],   val_df[args.label_col],   tok, args.max_length)
    test_ds  = HeadlineDS(test_df[args.text_col],  test_df[args.label_col],  tok, args.max_length)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained, num_labels=2
    )

    collator = DataCollatorWithPadding(
        tokenizer=tok,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )
    # W&B routing

    wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=os.environ.get("WANDB_RUN_NAME"),
        config=vars(args)
    )

    ta = TrainingArguments(
        output_dir=args.output_dir,
        run_name=(os.environ.get("WANDB_RUN_NAME") or args.wandb_run_name or f"{args.model_type}-{args.pretrained}"),
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
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
        seed=args.seed,
        report_to="wandb",
        no_cuda=(device != "cuda")
    )

    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    if device == "mps":
        trainer.model.to("mps")

    # Train
    trainer.train()

    # Final evaluations
    print("\n[Transformer] VAL:")
    val_metrics = trainer.evaluate(val_ds)
    print({k: round(float(v), 4) for k, v in val_metrics.items() if k.startswith(("eval_",))})

    print("\n[Transformer] TEST:")
    test_output = trainer.predict(test_ds)
    test_metrics = test_output.metrics
    print({k: round(float(v), 4) for k, v in test_metrics.items() if k.startswith(("test_", "eval_"))})

        
    test_logits = test_output.predictions
    test_preds  = np.argmax(test_logits, axis=-1).astype(int)
    test_labels = test_output.label_ids.astype(int)
    wandb.log({ f"test/{k.replace('test_', '')}": float(v) for k, v in test_metrics.items() if k.startswith("test_") })
    wandb.log({
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=test_labels, preds=test_preds,
            class_names=["not_clickbait", "clickbait"]
        )
    })
