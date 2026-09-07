import matplotlib.pyplot as plt
import numpy as np

from finetuning import config as C

PALETTE = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#b83280"]


def _save(fig, name: str):
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"{name}.png", dpi=150, bbox_inches="tight")
    return fig


def loss_curves(log_history, name: str = "loss_curves"):
    tr = [(h["step"], h["loss"]) for h in log_history if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in log_history if "eval_loss" in h]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot([s for s, _ in tr], [v for _, v in tr], color=PALETTE[0], lw=1.3,
            label="training loss")
    if ev:
        ax.plot([s for s, _ in ev], [v for _, v in ev], "o-", color=PALETTE[1], lw=1.8,
                ms=6, label="validation loss")
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training and validation loss"); ax.legend(fontsize=9); ax.grid(alpha=.25)
    return _save(fig, name)


def eval_metrics(log_history, name: str = "eval_metrics"):
    ev = [h for h in log_history if "eval_loss" in h]
    if not ev:
        return None
    epochs = [h["epoch"] for h in ev]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, key in enumerate(["eval_accuracy", "eval_precision", "eval_recall", "eval_f1"]):
        if key in ev[0]:
            ax.plot(epochs, [h[key] for h in ev], "o-", color=PALETTE[i], lw=1.6,
                    label=key.replace("eval_", ""))
    ax.set_xlabel("epoch"); ax.set_ylabel("score"); ax.set_xticks(epochs)
    ax.set_title("Validation metrics by epoch"); ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=.25)
    return _save(fig, name)




def reliability(y_true, prob, n_bins: int = 15, title: str = "Reliability",
                name: str = "reliability"):
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    conf = np.where(prob >= .5, prob, 1 - prob)
    correct = ((prob >= .5).astype(int) == y_true)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(m.sum())
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], ls="--", color="#718096", lw=1, label="perfectly calibrated")
    ax.plot(xs, ys, "o-", color=PALETTE[0], lw=1.6, label="model")
    ax.set_xlabel("confidence claimed"); ax.set_ylabel("accuracy observed")
    ax.set_xlim(.4, 1.02); ax.set_ylim(0, 1.02)
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=.25)
    return _save(fig, name)


def threshold_sweep(sweep, name: str = "threshold_sweep"):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, col in enumerate(["precision", "recall", "f1", "accuracy"]):
        ax.plot(sweep.index, sweep[col], label=col, color=PALETTE[i], lw=1.6)
    ax.axvline(.5, color="#718096", ls=":", lw=1.2, label="default 0.5")
    ax.set_xlabel("decision threshold"); ax.set_ylabel("score")
    ax.set_title("Metrics vs threshold"); ax.legend(fontsize=8, ncol=5); ax.grid(alpha=.25)
    return _save(fig, name)


def by_source_bars(df, metric: str = "accuracy", name: str = "by_source"):
    sub = df.drop(index="ALL", errors="ignore")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(sub.index, sub[metric], color=PALETTE[:len(sub)])
    if "ALL" in df.index:
        ax.axhline(df.loc["ALL", metric], color="#e53e3e", ls="--", lw=1.4,
                   label=f"pooled ({df.loc['ALL', metric]:.3f})")
        ax.legend(fontsize=8)
    for i, v in enumerate(sub[metric]):
        ax.text(i, v + .01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_ylabel(metric)
    ax.set_title(f"{metric} by source corpus"); ax.grid(axis="y", alpha=.25)
    return _save(fig, name)


def ood_separation(in_scores, out_scores, auroc_value: float, name: str = "ood_separation"):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(min(in_scores.min(), out_scores.min()),
                       max(in_scores.max(), out_scores.max()), 60)
    ax.hist(in_scores, bins=bins, alpha=.65, color=PALETTE[0], label="in-domain (test)")
    ax.hist(out_scores, bins=bins, alpha=.65, color=PALETTE[1], label="out-of-domain (tweets)")
    ax.set_xlabel("Mahalanobis distance"); ax.set_ylabel("count")
    ax.set_title(f"OOD separation  (AUROC = {auroc_value:.3f})")
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    return _save(fig, name)
