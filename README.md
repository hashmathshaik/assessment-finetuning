# Claim detection

Decides whether a sentence contains something a fact-checker could verify.
Not whether it's true.

> "The unemployment rate fell to 3.5% last year." → claim
> "I think we should lower taxes." → not a claim

Live at **https://claim-detection.fly.dev**

```bash
curl -X POST https://claim-detection.fly.dev/v1/detect \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: demo-1' \
  -d '{"sentences":["The unemployment rate fell to 3.5% last year."]}'
```

## Results

BERT-base fine-tuned on a 12,997-sentence corpus — 8,316 rows used for fitting,
the rest held out for validation, calibration and test. 69 minutes on a MacBook
Air using Apple's MPS backend. No CUDA, no cloud GPU, no rented hardware.

| | accuracy | F1 |
|---|---|---|
| TF-IDF + logistic regression | 0.849 | 0.835 |
| **BERT-base fine-tuned** | **0.909** | **0.900** |

## It collapses out of domain

| on tweets | accuracy | F1 | says "claim" |
|---|---|---|---|
| BERT | 0.638 | 0.776 | 98% |
| answering "claim" to everything | 0.630 | 0.773 | 100% |

It doesn't degrade, it stops discriminating — and lands on the score of a
constant. F1 hides it, because recall going to 1.0 props F1 up while precision
falls to the base rate. Every table here carries an all-positive row for that
reason.

**The threshold helps, but nowhere near enough.** Sweeping it on calibrated
probabilities takes accuracy from 0.630 to 0.716 — but only at 0.95, where you
are demanding near-certainty, recall drops to 0.897 and it still calls 78% of
tweets a claim. F1 moves 0.773 → 0.799. Against 0.909 in domain, that is not a
fix, and the operating point it needs is one you would never actually ship.

![threshold](artifacts/figures/ood_threshold_sweep.png)

**Not unfamiliar text.** Mahalanobis distance over the embeddings gives AUROC
**0.47** — worse than chance. The model doesn't find tweets strange, which is
why nothing internal flags the failure.

![ood](artifacts/figures/ood_separation.png)

**It's the data.** AVeriTeC is 3,068 of the rows and *every one is positive*,
because it was built from claims already fact-checked. Source and label are
entangled. The released splits are `text,label` only, so this isn't visible —
`data.py` recovers provenance by exact-matching each row to its corpus,
12,996 of 12,997.

![by source](artifacts/figures/by_source_f1.png)

Read that AVeriTeC bar carefully: it has no negative rows, so precision is 1.0
for any model that predicts positive at all, and F1 is inflated by construction.
The comparison that means something is Claimbuster 0.825 against PoliClaim 0.778
— and that the ablated model still scores 0.937 accuracy on AVeriTeC having
never trained on it.

The TF-IDF baseline collapses identically, which rules out the architecture.

**Ablation.** Retraining without AVeriTeC moves tweets 0.638 → 0.662 and drops
"says claim" from 98.8% to 94.4%. Standard error on 911 rows is 1.6 points, so
that's ~1.5 SE — suggestive, not significant. It also shifts the positive rate
48% → 32%, so it doesn't isolate the cause. Part of it, not all of it.

## Confidence

| | ECE | accuracy |
|---|---|---|
| uncalibrated | 0.049 | 0.911 |
| temperature (T=1.45) | 0.039 | 0.911 |
| isotonic | 0.021 | 0.909 |

Accuracy in this table is at threshold 0.5, which is where temperature provably
changes nothing. The shipped threshold is 0.35, chosen on validation, which is
why the headline table reads 0.909.

Temperature divides both logits by the same number, so it can't flip a
prediction — accuracy is identical. It makes the number honest, not the model
better. The API serves the calibrated probability.

**On tweets ECE is 0.31**: even after calibration it claims 95% confidence and is
right 64% of the time. Uncalibrated it was 0.34.

## Serving

![architecture](artifacts/figures/architecture.svg)

`/v1/detect` answers inline — 52ms p50 locally, 90–180ms on Fly. `/v1/jobs` returns a ticket for work that
outlives an HTTP connection.

**Both publish to NATS before running the model.** That ordering is the whole
guarantee — if the container dies mid-inference the message is still in the
stream and a worker finishes it.

The claim is narrower than "nothing is ever lost": *once we return a status
code, the work is durable.* Before that the client owns the retry, which is why
the idempotency key is client-generated.

**Crash recovery is free, hangs still cost you a timeout.** A worker holding an
unacked message over a live connection gets its work redelivered the moment that
connection drops — no code, no configuration. A worker that hangs while still
connected is a different problem, and NATS answers it the same way everything
does: `ack_wait` (300s here) plus `in_progress()` heartbeats every 20s so a
legitimately long batch keeps its claim. So the broker removes the crash case,
not the timeout you have to pick. Doing this in Postgres means writing both.

Results go in a KV bucket keyed by the idempotency key, so a retry returns the
stored answer.

Requests arriving within 5ms are batched into one forward pass.

The deployed endpoint is open on purpose — `ALLOW_ANONYMOUS=1` in `fly.toml` so
the curl above works. The service refuses to start if neither that nor `API_KEY`
is set, so auth can't end up disabled by accident.

### Measured

Killing workers mid-flight: **500 accepted, 500 completed, 0 lost.**

```
c=1     18 req/s   p50  52ms
c=32   100 req/s   p50 301ms   ← knee
c=64   108 req/s   p50 490ms
```

Zero errors at every level. Measured against uvicorn directly, not through
nginx — nginx caps a single IP at 50 r/s, so a load test pointed at it measures
the rate limiter rather than the service.

## Running it

```bash
python -m finetuning.train --model bert --max-length 64 --epochs 3   # once, ~69 min
docker compose up --build
```

The model weights are 418MB and aren't in git, so the build needs them present
at `artifacts/models/bert-seed42/`. Train once and they're there. To skip that,
hit the live URL above — it has them baked into the image.

```bash
python -m finetuning.train --model bert --max-length 64 --epochs 3
python -m finetuning.export
```

Training dumps logits and embeddings per split, so analysis never costs a
retrain. Notebooks: `01_train` for data and training, `02_analysis` for
everything above. Deployment: [deployment/DEPLOY.md](deployment/DEPLOY.md).

## What's missing

- **One NATS node is a single point of failure.** If it's down the API can't
  accept anything, because accepting means writing to it. `replicas=3` fixes it.
- Ablation doesn't isolate its variable — should downsample, not drop.
- One seed. Nothing has error bars.
- fp32 torch on CPU; ONNX INT8 would cut latency.

## Reference

Bell, A. (2025). *Less Can be More.* FEVER workshop, ACL. Splits from
[VeritaResearch/claim-extraction](https://github.com/VeritaResearch/claim-extraction).
