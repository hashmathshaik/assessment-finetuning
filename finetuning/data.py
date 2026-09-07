import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from finetuning import config as C


def _fetch(spec: tuple) -> "C.Path":
    url, relpath = spec
    dest = C.DATA_RAW / relpath
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def load_splits() -> "tuple[pd.DataFrame, pd.DataFrame]":
    train = pd.read_csv(_fetch(C.SPLIT_FILES["train"]))
    test = pd.read_csv(_fetch(C.SPLIT_FILES["test"]))
    return train, test


def load_ood() -> pd.DataFrame:
    path = _fetch(C.OOD_FILE)
    df = pd.read_csv(path, sep="\t")
    return df.rename(columns={"tweet_text": "text", "class_label": "label"})[["text", "label"]]


def load_sources() -> pd.DataFrame:
    frames = []


    path = _fetch(C.SOURCE_FILES["AVeriTeC"][0])
    av = pd.read_json(path).filter(items=["claim"]).rename(columns={"claim": "text"})
    av["label"] = 1
    av["source"] = "AVeriTeC"
    frames.append(av)


    poli = [pd.read_excel(_fetch(spec)) for spec in C.SOURCE_FILES["PoliClaim"]]
    pc = pd.concat(poli).filter(items=["SENTENCES", "golden"])
    pc = pc.rename(columns={"SENTENCES": "text", "golden": "label"})
    pc["source"] = "PoliClaim"
    frames.append(pc)


    path = _fetch(C.SOURCE_FILES["Claimbuster"][0])
    cb = pd.read_json(path).filter(items=["text", "label"])
    cb["source"] = "Claimbuster"
    frames.append(cb)

    return pd.concat(frames, ignore_index=True)


def attach_source(split: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    lookup = (sources.groupby("text")["source"]
              .agg(lambda s: s.iloc[0] if s.nunique() == 1 else "ambiguous"))
    out = split.copy()
    out["source"] = out["text"].map(lookup).fillna("unmatched")
    return out


def join_report(df: pd.DataFrame) -> pd.DataFrame:
    rep = df["source"].value_counts().rename("n").to_frame()
    rep["pct"] = (100 * rep["n"] / len(df)).round(2)
    return rep


def source_summary(sources: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, sub in list(sources.groupby("source")) + [("Total", sources)]:
        rows.append(dict(source=name, n=len(sub),
                         pct_positive=round(100 * sub["label"].mean(), 2)))
    return pd.DataFrame(rows).set_index("source")


def make_splits(train: pd.DataFrame, seed: int = C.SEED, val_frac: float = C.VAL_FRACTION,
                cal_frac: float = C.CAL_FRACTION) -> tuple:
    rest, val = train_test_split(train, test_size=val_frac, random_state=seed,
                                 stratify=train["label"])
    fit, cal = train_test_split(rest, test_size=cal_frac / (1 - val_frac),
                                random_state=seed, stratify=rest["label"])
    return (fit.reset_index(drop=True), val.reset_index(drop=True),
            cal.reset_index(drop=True))
