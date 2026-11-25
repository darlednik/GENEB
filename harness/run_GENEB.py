#!/usr/bin/env python3
"""
GENEB reference harness: run an extractor on the pinned task set and write a submission file.

Loads a class from harness/extractors/, embeds train/test sequences for each task, fits
the protocol-defined logistic-regression probe (seeds and settings from benchmark_spec.json)
in regimes full / 1shot / 10shot, and writes submissions/<model_id>.json.

Example:

    python3 harness/run_GENEB.py \\
        --extractor MyExtractor --module my_model \\
        --name_model org/my-model \\
        --model_id my-model-300m --display "My Model 300M" --params 300000000 \\
        --url https://huggingface.co/org/my-model \\
        --data_dir ./GENEB_data --device cuda --submitted_by "Your Name"

Dataset: one CSV per task named <task_id>.csv; columns text, label, split (train|test).
"""
import argparse, importlib, json, os, sys, csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)  # allow `import extractors.<module>`
SPEC = json.load(open(os.path.join(ROOT, "benchmark", "benchmark_spec.json")))

TASKS    = [t["id"] for t in SPEC["tasks"]]
REGIMES  = SPEC["regimes"]                      # ["full","10shot","1shot"]
SEEDS    = SPEC["protocol"]["seeds"]            # [13,17,42,123,997]
LOGREG   = SPEC["protocol"]["logreg"]
MAX_ITER = LOGREG["max_iter"]
LOGREG_KW = {"max_iter": MAX_ITER}
if "n_jobs" in LOGREG:
    LOGREG_KW["n_jobs"] = LOGREG["n_jobs"]
# The probe usually stops at max_iter, so its result depends on BLAS summation order.
# Pinning the thread count keeps scores independent of the host's core count.
BLAS_THREADS = LOGREG.get("blas_threads")  # None leaves BLAS threading unchanged
KSHOTS   = SPEC["protocol"]["few_shot_k"]       # [1,10]
K_TO_REG = {1: "1shot", 10: "10shot"}


def load_extractor(module, cls, name_model, device):
    mod = importlib.import_module(f"extractors.{module}")
    return getattr(mod, cls)(name_model=name_model, device=device)


def read_task_csv(path):
    rows = list(csv.DictReader(open(path)))
    need = {"text", "label", "split"}
    if not need.issubset(rows[0].keys()):
        raise ValueError(f"{path}: columns must be text,label,split (got {list(rows[0].keys())})")
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    return ([r["text"] for r in tr], np.array([int(r["label"]) for r in tr]),
            [r["text"] for r in te], np.array([int(r["label"]) for r in te]))


def metrics(y, p):
    return (matthews_corrcoef(y, p), accuracy_score(y, p), f1_score(y, p, average="macro"))


def fit_eval(Xtr, ytr, Xte, yte, idx=None):
    """Mean MCC/Acc/F1 over seeds; idx (per-seed) selects a few-shot subset."""
    mc, ac, f1 = [], [], []
    for seed in SEEDS:
        if idx is None:
            xs, ys = Xtr, ytr
        else:
            sel = idx(seed)
            xs, ys = Xtr[sel], ytr[sel]
        with threadpool_limits(limits=BLAS_THREADS, user_api="blas"):
            clf = LogisticRegression(**LOGREG_KW, random_state=seed).fit(xs, ys)
            m, a, f = metrics(yte, clf.predict(Xte))
        mc.append(m); ac.append(a); f1.append(f)
    return {"MCC": float(np.mean(mc)), "Acc": float(np.mean(ac)), "F1": float(np.mean(f1))}


def few_shot_idx(ytr, k):
    def pick(seed):
        rng = np.random.RandomState(seed)
        return np.concatenate([
            rng.choice(locs, size=min(k, len(locs)), replace=False)
            for cls in np.unique(ytr)
            for locs in [np.where(ytr == cls)[0]]
        ])
    return pick


def evaluate_task(extractor, batch_size, Xtr_seq, ytr, Xte_seq, yte):
    Etr = extractor.extract_embeddings(Xtr_seq, batch_size)
    Ete = extractor.extract_embeddings(Xte_seq, batch_size)
    out = {"full": fit_eval(Etr, ytr, Ete, yte)}
    for k in KSHOTS:
        out[K_TO_REG[k]] = fit_eval(Etr, ytr, Ete, yte, idx=few_shot_idx(ytr, k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", required=True)
    ap.add_argument("--module", required=True)
    ap.add_argument("--name_model", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--display", required=True)
    ap.add_argument("--params", type=int, required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--url", default="")
    ap.add_argument("--submitted_by", default="")
    ap.add_argument("--training_data", default="")
    ap.add_argument("--zero_shot", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N tasks (subset run)")
    a = ap.parse_args()

    # require a complete local task set before embedding extraction
    present = {t for t in TASKS if os.path.exists(os.path.join(a.data_dir, t + ".csv"))}
    missing = set(TASKS) - present
    todo = (TASKS[:a.limit] if a.limit else TASKS)
    if missing and not a.limit:
        print(f"ERROR: dataset is missing {len(missing)} task CSV(s), e.g. {sorted(missing)[:3]}")
        print(f"       expected <task_id>.csv for each task in benchmark_spec.json under {a.data_dir}")
        sys.exit(2)

    extractor = load_extractor(a.module, a.extractor, a.name_model, a.device)
    results = {r: {} for r in REGIMES}
    for i, tid in enumerate(todo, 1):
        csv_path = os.path.join(a.data_dir, tid + ".csv")
        if not os.path.exists(csv_path):
            print(f"  [{i}/{len(todo)}] skip {tid} (no csv)"); continue
        Xtr, ytr, Xte, yte = read_task_csv(csv_path)
        print(f"  [{i}/{len(todo)}] {tid}  (train={len(Xtr)} test={len(Xte)})", flush=True)
        per = evaluate_task(extractor, a.batch_size, Xtr, ytr, Xte, yte)
        for reg in REGIMES:
            results[reg][tid] = per[reg]

    sub = {
        "schema_version": "1.0",
        "model_id": a.model_id,
        "harness_version": SPEC.get("harness_version", "GENEB-0.1.0"),
        "meta": {"display": a.display, "params": a.params, "url": a.url,
                 "zero_shot": a.zero_shot, "training_data": a.training_data,
                 "provenance": "self-reported", "submitted_by": a.submitted_by},
        "results": {reg: {t: results[reg][t] for t in TASKS if t in results[reg]} for reg in REGIMES},
    }
    import re
    out = os.path.join(ROOT, "submissions", re.sub(r"[^A-Za-z0-9._-]", "_", a.model_id) + ".json")
    json.dump(sub, open(out, "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"\nwrote {out}")
    if a.limit:
        print(f"(subset run: {a.limit}/{len(TASKS)} tasks; full benchmark requires all tasks)")
    else:
        print("validate: python tools/validate_submission.py " + os.path.relpath(out, ROOT))
        print("submit: submissions file, harness/extractors/%s.py, model_cards/%s.md"
              % (a.module, a.model_id))


if __name__ == "__main__":
    main()
