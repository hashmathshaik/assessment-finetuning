import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "artifacts/models/bert-seed42"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "64"))
MAX_SENTENCES = int(os.environ.get("MAX_SENTENCES", "256"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "2000"))


class Detector:
    def __init__(self, model_dir: Path = MODEL_DIR):
        cfg = json.loads((model_dir / "serving.json").read_text())
        self.temperature = cfg["temperature"]
        self.threshold = cfg["threshold"]
        self.version = cfg["version"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

    @torch.no_grad()
    def predict(self, sentences: list[str]) -> list[dict]:
        batch = self.tokenizer(sentences, truncation=True, max_length=MAX_LENGTH,
                               padding=True, return_tensors="pt")
        logits = self.model(**batch).logits.numpy() / self.temperature
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        prob = (e / e.sum(axis=1, keepdims=True))[:, 1]
        out = []
        for p in prob:
            is_claim = bool(p >= self.threshold)
            out.append(dict(is_claim=is_claim,
                            confidence=round(float(p if is_claim else 1 - p), 4)))
        return out
