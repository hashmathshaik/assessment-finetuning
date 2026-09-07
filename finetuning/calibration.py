import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from finetuning import metrics as M


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def positive_prob(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return softmax(logits / temperature)[:, 1]


def _nll(temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(softmax(logits / temperature), 1e-12, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(labels)), labels])))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    res = minimize_scalar(_nll, bounds=(0.05, 20.0), args=(logits, labels.astype(int)),
                          method="bounded")
    return float(res.x)




def fit_platt(logits: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    z = (logits[:, 1] - logits[:, 0]).reshape(-1, 1)
    return LogisticRegression(C=1e6, solver="lbfgs").fit(z, labels.astype(int))


def apply_platt(model: LogisticRegression, logits: np.ndarray) -> np.ndarray:
    z = (logits[:, 1] - logits[:, 0]).reshape(-1, 1)
    return model.predict_proba(z)[:, 1]


def fit_isotonic(logits: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip").fit(positive_prob(logits),
                                                        labels.astype(int))


def apply_isotonic(model: IsotonicRegression, logits: np.ndarray) -> np.ndarray:
    return model.predict(positive_prob(logits))


def pick_threshold(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0,
                   metric: str = "f1") -> float:
    prob = positive_prob(logits, temperature)
    sweep = M.sweep_threshold(labels, prob, np.arange(0.05, 0.96, 0.01))
    return float(sweep[metric].idxmax())


def compare(cal_logits: np.ndarray, cal_labels: np.ndarray,
            targets: dict, n_bins: int = 15) -> pd.DataFrame:
    T = fit_temperature(cal_logits, cal_labels)
    platt = fit_platt(cal_logits, cal_labels)
    iso = fit_isotonic(cal_logits, cal_labels)

    methods = {
        "uncalibrated": lambda lg: positive_prob(lg),
        f"temperature (T={T:.3f})": lambda lg: positive_prob(lg, T),
        "platt": lambda lg: apply_platt(platt, lg),
        "isotonic": lambda lg: apply_isotonic(iso, lg),
    }

    rows = []
    for split, (lg, y) in targets.items():
        for name, fn in methods.items():
            p = fn(lg)
            rows.append(dict(split=split, method=name,
                             ece=M.ece(y, p, n_bins), brier=M.brier(y, p),
                             mean_confidence=float(np.mean(np.maximum(p, 1 - p))),
                             accuracy=float(np.mean((p >= 0.5).astype(int) == y))))
    return pd.DataFrame(rows).set_index(["split", "method"]).round(4)
