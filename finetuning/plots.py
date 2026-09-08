import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from finetuning import config as C

BLUE, ORANGE, GREEN, PURPLE, GREY = "#2f6fb2", "#d2691e", "#2e8b57", "#7b52ab", "#9aa5b1"
RED = "#c0392b"

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.labelsize": 11,
    "axes.labelcolor": "#3d4852",
    "axes.edgecolor": "#c9d1d9",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#3d4852",
    "ytick.color": "#3d4852",
    "grid.color": "#e6e9ed",
    "grid.linewidth": 0.9,
    "legend.frameon": False,
})


def _save(fig, name: str):
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"{name}.png", bbox_inches="tight", facecolor="white")
    return fig


def loss_curves(log_history, name: str = "loss_curves"):
    tr = [(h["step"], h["loss"]) for h in log_history if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in log_history if "eval_loss" in h]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([s for s, _ in tr], [v for _, v in tr], color=BLUE, lw=1.6, label="training")
    if ev:
        ax.plot([s for s, _ in ev], [v for _, v in ev], "o-", color=ORANGE, lw=2.2,
                ms=7, label="validation")
        best = min(ev, key=lambda t: t[1])
        ax.annotate("validation stops improving here",
                    xy=best, xytext=(best[0] + 120, best[1] + 0.12),
                    color="#3d4852", fontsize=10,
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training and validation loss")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=.7)
    return _save(fig, name)


def eval_metrics(log_history, name: str = "eval_metrics"):
    ev = [h for h in log_history if "eval_loss" in h]
    if not ev:
        return None
    ep = [h["epoch"] for h in ev]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for c, k in zip([BLUE, ORANGE, GREEN, PURPLE],
                    ["eval_accuracy", "eval_precision", "eval_recall", "eval_f1"]):
        if k in ev[0]:
            ax.plot(ep, [h[k] for h in ev], "o-", color=c, lw=2, ms=6,
                    label=k.replace("eval_", ""))
    ax.set_xlabel("epoch"); ax.set_ylabel("score"); ax.set_xticks(ep)
    ax.set_title("Validation metrics by epoch")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(axis="y", alpha=.7)
    return _save(fig, name)


def reliability(y_true, prob, n_bins: int = 15, title: str = "Reliability",
                name: str = "reliability"):
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    conf = np.where(prob >= .5, prob, 1 - prob)
    ok = ((prob >= .5).astype(int) == y_true)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            xs.append(conf[m].mean()); ys.append(ok[m].mean()); ns.append(m.sum())
    lo_lim = min(0.45, min(ys) - 0.05)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1.3, label="perfect")
    ax.plot(xs, ys, "-", color=BLUE, lw=2, zorder=3)
    smax = max(ns)
    ax.scatter(xs, ys, s=[30 + 220 * (n / smax) for n in ns], color=BLUE,
               alpha=.75, zorder=4, label="model (size = bin count)")
    ax.fill_between(xs, ys, xs, color=RED, alpha=.10)
    ax.set_xlabel("confidence claimed"); ax.set_ylabel("accuracy observed")
    ax.set_xlim(lo_lim, 1.02); ax.set_ylim(lo_lim, 1.02)
    ax.set_title(title); ax.legend(loc="upper left", fontsize=9.5); ax.grid(alpha=.7)
    ax.set_aspect("equal", adjustable="box")
    ax.text(0, -.15, "below the dashed line means overconfident",
            transform=ax.transAxes, fontsize=9.5, color="#57606a")
    return _save(fig, name)


def threshold_sweep(sweep, baseline: "float | None" = None, name: str = "threshold_sweep"):
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for c, k in zip([PURPLE, BLUE, ORANGE, GREEN],
                    ["accuracy", "precision", "recall", "f1"]):
        ax.plot(sweep.index, sweep[k], color=c, lw=2.2, label=k)
    if baseline is not None:
        ax.axhline(baseline, color=RED, ls="--", lw=1.4)
        ax.text(sweep.index.min(), baseline + .008, "all-positive baseline",
                color=RED, fontsize=9.5, va="bottom")
    ax.axvline(.5, color=GREY, ls=":", lw=1.3)
    ax.text(.5, ax.get_ylim()[0], " default 0.5", color="#57606a", fontsize=9.5, va="bottom")
    ax.set_xlabel("decision threshold"); ax.set_ylabel("score")
    ax.set_title("Out-of-domain metrics across every threshold")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(axis="y", alpha=.7)
    return _save(fig, name)


def by_source_bars(df, metric: str = "accuracy", name: str = "by_source", note: str = ""):
    sub = df.drop(index="ALL", errors="ignore").sort_values(metric)
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    colors = [ORANGE if i == "AVeriTeC" else BLUE for i in sub.index]
    ax.barh(sub.index, sub[metric], color=colors, height=.6, zorder=3)
    for i, (src, v) in enumerate(sub[metric].items()):
        ax.text(v + .012, i, f"{v:.3f}", va="center", fontsize=11)
        if src == "AVeriTeC":
            ax.text(v / 2, i, "every row is positive, so precision is 1.0 for free",
                    va="center", ha="center", fontsize=9.5, color="white")
    if "ALL" in df.index:
        pooled = df.loc["ALL", metric]
        ax.axvline(pooled, color=RED, ls="--", lw=1.5, zorder=4)
        ax.text(pooled, len(sub) - .35, f"  pooled {pooled:.3f}", color=RED,
                fontsize=9.5, va="bottom")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(metric)
    ax.set_title(f"{metric} by source corpus")
    ax.grid(axis="x", alpha=.7, zorder=0)
    if note:
        fig.text(0.01, -0.06, note, fontsize=9.5, color="#57606a")
    return _save(fig, name)


def ood_separation(in_scores, out_scores, auroc_value: float, name: str = "ood_separation"):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(min(in_scores.min(), out_scores.min()),
                       max(in_scores.max(), out_scores.max()), 55)
    ax.hist(in_scores, bins=bins, density=True, alpha=.55, color=BLUE,
            label="in-domain (test)")
    ax.hist(out_scores, bins=bins, density=True, alpha=.55, color=ORANGE,
            label="out-of-domain (tweets)")
    ax.set_xlabel("Mahalanobis distance from training data")
    ax.set_ylabel("density")
    ax.set_title(f"The model cannot tell tweets apart   ·   AUROC {auroc_value:.3f}")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=.7)
    ax.text(0, -.22, "0.5 means no separation at all", transform=ax.transAxes,
            fontsize=9.5, color="#57606a")
    return _save(fig, name)
