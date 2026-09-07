import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def score(y_true, y_pred) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                  zero_division=0)
    return dict(accuracy=accuracy_score(y_true, y_pred), precision=p, recall=r, f1=f1,
                n=len(y_true), pred_positive_rate=float(np.mean(y_pred)))


def always_positive(y_true) -> dict:
    return score(y_true, np.ones_like(np.asarray(y_true)))


def table(y_true, preds: dict) -> pd.DataFrame:
    rows = {name: score(y_true, yp) for name, yp in preds.items()}
    rows["all-positive baseline"] = always_positive(y_true)
    df = pd.DataFrame(rows).T
    return df.round(4)


def by_source(df: pd.DataFrame, y_pred, label_col: str = "label") -> pd.DataFrame:
    out = df.copy()
    out["pred"] = np.asarray(y_pred)
    rows = {}
    for src, g in out.groupby("source"):
        s = score(g[label_col], g["pred"])
        s["true_positive_rate"] = float(g[label_col].mean())
        rows[src] = s
    rows["ALL"] = score(out[label_col], out["pred"])
    rows["ALL"]["true_positive_rate"] = float(out[label_col].mean())
    return pd.DataFrame(rows).T.round(4)


def sweep_threshold(y_true, prob, grid=None) -> pd.DataFrame:
    grid = np.arange(0.05, 0.96, 0.05) if grid is None else grid
    return pd.DataFrame([dict(threshold=round(t, 2), **score(y_true, (prob >= t).astype(int)))
                         for t in grid]).set_index("threshold").round(4)


def ece(y_true, prob, n_bins: int = 15) -> float:
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    conf = np.where(prob >= 0.5, prob, 1 - prob)
    correct = (prob >= 0.5).astype(int) == y_true
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            total += (m.sum() / len(prob)) * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def auroc(y_true, prob) -> float:
    return float(roc_auc_score(y_true, prob))


def brier(y_true, prob) -> float:
    return float(np.mean((np.asarray(prob) - np.asarray(y_true)) ** 2))
