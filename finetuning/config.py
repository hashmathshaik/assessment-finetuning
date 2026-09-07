from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "claim-extraction"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "artifacts" / "models"
FIGURES = ROOT / "artifacts" / "figures"
RESULTS = ROOT / "artifacts" / "results"

for _p in (DATA_RAW, DATA_PROC, MODELS, FIGURES, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)


PAPER_REPO = "https://raw.githubusercontent.com/VeritaResearch/claim-extraction/main"


SPLIT_FILES = {
    "train": (f"{PAPER_REPO}/data/ours/train.csv", "splits/train.csv"),
    "test": (f"{PAPER_REPO}/data/ours/test.csv", "splits/test.csv"),
}


OOD_FILE = (f"{PAPER_REPO}/data/CheckThat/CT22_english_1B_claim_dev_test.tsv",
            "checkthat/ct22_english_1B.tsv")


POLICLAIM_SHEETS = ["AL2003_G4_1", "CT2014_G4_1", "DE1999_G4_1", "DE2021_G4_1",
                    "IN2001_G4_1", "IN2011_G4_1", "KY2018_G4_1", "US2016_G4_1"]

SOURCE_FILES = {
    "AVeriTeC": [(f"{PAPER_REPO}/data/AVeriTeC/train.json", "averitec/train.json")],
    "Claimbuster": [(f"{PAPER_REPO}/data/Claimbuster/full.json", "claimbuster/full.json")],
    "PoliClaim": [(f"{PAPER_REPO}/data/PoliClaim/{n}.xlsx", f"policlaim/{n}.xlsx")
                  for n in POLICLAIM_SHEETS],
}


SEED = 42


VAL_FRACTION = 0.10
CAL_FRACTION = 0.10

MODELS_TO_TRAIN = {
    "bert": "google-bert/bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "minilm": "microsoft/MiniLM-L6-H384-uncased",
}

HPARAMS = dict(
    max_length=128,
    batch_size=32,
    eval_batch_size=128,
    learning_rate=2e-5,
    weight_decay=0.01,
    epochs=5,
    warmup_ratio=0.06,
    fp16=True,
)
