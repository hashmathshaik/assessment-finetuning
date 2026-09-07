import numpy as np

from finetuning import metrics as M


class Mahalanobis:
    def __init__(self, shrinkage: float = 1e-3):
        self.shrinkage = shrinkage
        self.mean = None
        self.precision = None

    def fit(self, embeddings: np.ndarray) -> "Mahalanobis":
        x = np.asarray(embeddings, dtype=np.float64)
        self.mean = x.mean(axis=0)
        cov = np.cov(x - self.mean, rowvar=False)
        cov += self.shrinkage * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        self.precision = np.linalg.inv(cov)
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        d = np.asarray(embeddings, dtype=np.float64) - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.precision, d))

    def nbytes(self) -> int:
        return self.mean.nbytes + self.precision.nbytes


def separation_auroc(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    y = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
    return M.auroc(y, np.concatenate([in_scores, out_scores]))
