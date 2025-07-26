import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import logging

class NucleotideTransformerEmbeddingExtractor:
    def __init__(self,
                 name_model: str,
                 device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(name_model, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(name_model, trust_remote_code=True)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = self.tokenizer.model_max_length

    def extract_embeddings(self,
                           sequences: list[str],
                           batch_size: int = 8) -> np.ndarray:
        all_embs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc='Extracting embs...'):
                batch = sequences[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                hidden = outputs.hidden_states[-1]
                mask = attention_mask.unsqueeze(-1)
                sum_hidden = (hidden * mask).sum(dim=1)
                lengths = mask.sum(dim=1)
                seq_emb = (sum_hidden / lengths).cpu().numpy()
                all_embs.append(seq_emb)
        return np.vstack(all_embs)

def main():
    parser = argparse.ArgumentParser(
        description="Exctracting embeddings and training logreg.")
    parser.add_argument('-b','--batch_size', type=int, default=32)
    parser.add_argument('-d','--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('-o','--output_dir', type=str, default='.')
    parser.add_argument('-l','--log', action='store_true')
    parser.add_argument('--models', nargs='+', type=str, default=[
        'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species',
        'InstaDeepAI/nucleotide-transformer-v2-250m-multi-species',
        'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species',
        'InstaDeepAI/nucleotide-transformer-2.5b-multi-species'
    ], help='list models')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.log else logging.WARNING,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )
    logger = logging.getLogger()


    logger.info(f"Device: {args.device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset...")

    ds = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks")
    train_ds, test_ds = ds["train"], ds["test"]

    logger.info(f"Dataset is loaded.")

    PARAMS_LOGREG = {"max_iter": 1000, "random_state": 42}

    for name_model in args.models:
        logger.info(f"Model: {name_model}")
        extractor = NucleotideTransformerEmbeddingExtractor(name_model=name_model,
                                                           device=args.device)
        
        logger.info(f"Embedder is ready!")

        logger.info(f"Work starts with the full data....")

        # --- Baseline ---

        baseline = {}
        for task in tqdm(set(train_ds["task"]), desc="Baseline"):
            tr = train_ds.filter(lambda x, t=task: x["task"] == t)
            te = test_ds.filter(lambda x, t=task: x["task"] == t)
            seqs_tr, y_tr = tr["sequence"], np.array(tr["label"])
            seqs_te, y_te = te["sequence"], np.array(te["label"])

            X_tr = extractor.extract_embeddings(seqs_tr, batch_size=args.batch_size)
            X_te = extractor.extract_embeddings(seqs_te, batch_size=args.batch_size)

            clf = LogisticRegression(**PARAMS_LOGREG)
            clf.fit(X_tr.reshape(len(X_tr), -1), y_tr)
            preds = clf.predict(X_te.reshape(len(X_te), -1))

            baseline[task] = {"accuracy": float(accuracy_score(y_te, preds)),
                              "f1_score": float(f1_score(y_te, preds, average="macro"))}
            with open(out_dir / f"baseline_{name_model.split('/')[-1]}_{task}.json", "w") as f:
                json.dump(baseline, f, indent=4)
        
        logger.info(f"The work with full data has been successfully completed!")
        logger.info(f"Few-shot starts...")

        # --- Few-shot ---
        def few_shot(train, test, ks=[1,5,10,20], trials=5):
            res = {}
            rng = np.random.RandomState(42)
            for task in tqdm(set(train["task"]), desc="Few-shot"):
                tr = train.filter(lambda x, t=task: x["task"] == t)
                te = test.filter(lambda x, t=task: x["task"] == t)
                seqs_tr, y_tr = tr["sequence"], np.array(tr["label"])
                seqs_te, y_te = te["sequence"], np.array(te["label"])
                X_te = extractor.extract_embeddings(seqs_te, batch_size=args.batch_size)
                res[task] = {}
                for k in ks:
                    accs, f1s = [], []
                    for _ in range(trials):
                        idxs = []
                        for lbl in np.unique(y_tr):
                            locs = np.where(y_tr==lbl)[0]
                            choice = rng.choice(locs, size=min(k,len(locs)), replace=False)
                            idxs.extend(choice.tolist())
                        X_k = extractor.extract_embeddings([seqs_tr[i] for i in idxs],
                                                          batch_size=args.batch_size)
                        clf = LogisticRegression(**PARAMS_LOGREG)
                        clf.fit(X_k.reshape(len(X_k), -1), y_tr[idxs])
                        p = clf.predict(X_te.reshape(len(X_te), -1))
                        accs.append(accuracy_score(y_te,p))
                        f1s.append(f1_score(y_te,p,average="macro"))
                    res[task][k] = {"accuracy": float(np.mean(accs)),
                                   "f1_score": float(np.mean(f1s))}
                    with open(out_dir / f"kshot_{name_model.split('/')[-1]}_{task}_{k}.json", "w") as f:
                        json.dump(res, f, indent=4)
            return res

        logger.info(f"Few-shot has been successfully completed!")
        logger.info(f"Saving full results in JSON format...")

        results_kshot = few_shot(train_ds, test_ds)
        summary = {"full": baseline, "kshot": results_kshot, "params": PARAMS_LOGREG}
        with open(out_dir / f"results_summary_{name_model.split('/')[-1]}.json", "w") as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Finished {name_model}")

    logger.info("All done.")
    
if __name__ == "__main__":
    main()
