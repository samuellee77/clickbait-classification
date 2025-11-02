# dataset.py
import re
import pandas as pd

# Lowercase; remove digits & most punctuation, KEEP '!' and '?'
_CLEAN_RE = re.compile(r"[^a-zA-Z!? \t]+")

def clean_text_keep_bang_qmark(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.lower()
    text = _CLEAN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def load_csv_safely(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        engine="python",
        quotechar='"',
        doublequote=True,
        escapechar="\\",
        on_bad_lines="skip"
    )
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Expected columns '{text_col}' and '{label_col}' in {path}. Found {df.columns.tolist()}")
    df[label_col] = df[label_col].astype(str).str.strip().replace({"0":"0","1":"1"}).astype(int)
    return df[[text_col, label_col]].dropna()
