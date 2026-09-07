import argparse
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                          TrainingArguments)

from finetuning import config as C
from finetuning import data as D
from finetuning import metrics as M


class Sentences(Dataset):
    def __init__(self, texts, labels, tok, max_length: int):
        self.enc = tok(list(texts), truncation=True, max_length=max_length, padding=False)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def infer(model, tok, texts, max_length: int, batch_size: int, device: str):
    model.eval()
    logits, embs = [], []
    for i in range(0, len(texts), batch_size):
        batch = tok(list(texts[i:i + batch_size]), truncation=True, max_length=max_length,
                    padding=True, return_tensors="pt").to(device)
        out = model(**batch, output_hidden_states=True)
        logits.append(out.logits.float().cpu().numpy())
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (out.hidden_states[-1] * mask).sum(1) / mask.sum(1)
        embs.append(pooled.float().cpu().numpy())
    return np.vstack(logits), np.vstack(embs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert", choices=list(C.MODELS_TO_TRAIN))
    ap.add_argument("--epochs", type=int, default=C.HPARAMS["epochs"])
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--max-length", type=int, default=C.HPARAMS["max_length"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--exclude-source", default=None,
                    help="drop a corpus from training, e.g. AVeriTeC")
    ap.add_argument("--dump-only", action="store_true",
                    help="reload the saved model and rewrite the .npz arrays, no training")
    args = ap.parse_args()

    checkpoint = C.MODELS_TO_TRAIN[args.model]
    device = pick_device()
    hp = {**C.HPARAMS, "epochs": args.epochs, "max_length": args.max_length,
          "fp16": C.HPARAMS["fp16"] and device == "cuda"}
    if args.batch_size:
        hp["batch_size"] = args.batch_size
    if device == "mps":

        hp["batch_size"] = min(hp["batch_size"], 16)
        hp["eval_batch_size"] = min(hp["eval_batch_size"], 64)

    print(f"model={args.model} ({checkpoint})  device={device}  seed={args.seed}")
    print("hparams=" + json.dumps(hp))
    set_seed(args.seed)

    train_raw, test = D.load_splits()
    ood = D.load_ood()

    bad = train_raw["text"].isna().sum()
    if bad:
        print(f"dropping {bad} row(s) with null text from train")
        train_raw = train_raw.dropna(subset=["text"]).reset_index(drop=True)
    tag = args.model if not args.exclude_source else f"{args.model}-no{args.exclude_source}"
    if args.exclude_source:
        before = len(train_raw)
        train_raw = D.attach_source(train_raw, D.load_sources())
        train_raw = train_raw[train_raw["source"] != args.exclude_source]
        train_raw = train_raw[["text", "label"]].reset_index(drop=True)
        print(f"excluded {args.exclude_source}: {before} -> {len(train_raw)} rows, "
              f"{100 * train_raw['label'].mean():.2f}% positive")

    fit, val, cal = D.make_splits(train_raw, seed=args.seed)
    print(f"fit={len(fit)}  val={len(val)}  cal={len(cal)}  test={len(test)}  ood={len(ood)}")

    outdir = C.MODELS / f"{tag}-seed{args.seed}"
    load_from = str(outdir) if args.dump_only else checkpoint
    tok = AutoTokenizer.from_pretrained(load_from)
    model = AutoModelForSequenceClassification.from_pretrained(load_from, num_labels=2).to(device)


    steps_per_epoch = -(-len(fit) // hp["batch_size"])
    warmup_steps = int(hp["warmup_ratio"] * steps_per_epoch * hp["epochs"])
    targs = TrainingArguments(
        output_dir=str(outdir / "checkpoints"),
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["eval_batch_size"],
        learning_rate=hp["learning_rate"], weight_decay=hp["weight_decay"],
        num_train_epochs=hp["epochs"], warmup_steps=warmup_steps,
        fp16=hp["fp16"], seed=args.seed, logging_steps=25,
        eval_strategy="epoch", save_strategy="no", report_to=[],
    )
    trainer = Trainer(
        model=model, args=targs, processing_class=tok,
        train_dataset=Sentences(fit["text"], fit["label"], tok, hp["max_length"]),
        eval_dataset=Sentences(val["text"], val["label"], tok, hp["max_length"]),
        compute_metrics=lambda p: M.score(p.label_ids, p.predictions.argmax(-1)),
    )

    mins, final_loss = 0.0, None
    if not args.dump_only:
        t0 = time.time()
        hist = trainer.train()
        mins = (time.time() - t0) / 60
        final_loss = hist.training_loss
        print(f"trained in {mins:.1f} min  final_loss={final_loss:.4f}")
        outdir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(outdir); tok.save_pretrained(outdir)


    for name, df in [("test", test), ("ood", ood), ("cal", cal), ("val", val)]:
        lg, em = infer(model, tok, df["text"].tolist(), hp["max_length"],
                       hp["eval_batch_size"], device)
        np.savez_compressed(C.RESULTS / f"{tag}_{name}.npz",
                            logits=lg, embeddings=em, labels=df["label"].to_numpy())
        print(f"wrote {tag}_{name}.npz  logits={lg.shape} emb={em.shape}")


    lg, em = infer(model, tok, fit["text"].tolist(), hp["max_length"],
                   hp["eval_batch_size"], device)
    np.savez_compressed(C.RESULTS / f"{tag}_fit.npz",
                        logits=lg, embeddings=em, labels=fit["label"].to_numpy())

    (outdir / "run.json").write_text(json.dumps(
        dict(model=tag, checkpoint=checkpoint, excluded_source=args.exclude_source, seed=args.seed, device=device,
             hparams=hp, minutes=round(mins, 2), final_loss=final_loss,
             n_fit=len(fit), n_val=len(val), n_cal=len(cal),
             log_history=trainer.state.log_history,
             steps_per_epoch=steps_per_epoch), indent=2))
    print(f"done -> {outdir}")


if __name__ == "__main__":
    main()
