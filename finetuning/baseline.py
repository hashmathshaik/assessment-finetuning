import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from finetuning import config as C
from finetuning import data as D
from finetuning import metrics as M


def build() -> "make_pipeline":
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                        strip_accents="unicode", lowercase=True),
        LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
    )


def main() -> None:
    train, test = D.load_splits()
    train = train.dropna(subset=["text"]).reset_index(drop=True)
    ood = D.load_ood()
    sources = D.load_sources()

    clf = build().fit(train["text"], train["label"])

    out = {}
    for name, df in [("test (in-domain)", test), ("ood (tweets)", ood)]:
        prob = clf.predict_proba(df["text"])[:, 1]
        pred = (prob >= 0.5).astype(int)
        tbl = M.table(df["label"], {"tfidf+logreg": pred})
        print(f"\n=== {name} ===")
        print(tbl.to_string())
        out[name] = dict(scores=M.score(df["label"], pred), ece=M.ece(df["label"], prob),
                         auroc=M.auroc(df["label"], prob))


    test_s = D.attach_source(test, sources)
    pred = clf.predict(test["text"])
    print("\n=== by source (in-domain test) ===")
    print(M.by_source(test_s, pred).to_string())

    (C.RESULTS / "baseline_tfidf.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
