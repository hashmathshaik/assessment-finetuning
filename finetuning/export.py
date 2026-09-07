import argparse
import json

import numpy as np

from finetuning import calibration as CAL
from finetuning import config as C
from finetuning import metrics as M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    cal = np.load(C.RESULTS / f"{args.model}_cal.npz")
    val = np.load(C.RESULTS / f"{args.model}_val.npz")
    test = np.load(C.RESULTS / f"{args.model}_test.npz")

    temperature = CAL.fit_temperature(cal["logits"], cal["labels"])
    threshold = CAL.pick_threshold(val["logits"], val["labels"], temperature)

    prob = CAL.positive_prob(test["logits"], temperature)
    scores = M.score(test["labels"], (prob >= threshold).astype(int))

    outdir = C.MODELS / f"{args.model}-seed{args.seed}"
    cfg = dict(model=args.model, version=f"{args.model}-seed{args.seed}",
               temperature=round(temperature, 6), threshold=round(threshold, 4),
               test_accuracy=round(scores["accuracy"], 4),
               test_f1=round(scores["f1"], 4),
               test_ece=round(M.ece(test["labels"], prob), 4))
    (outdir / "serving.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
