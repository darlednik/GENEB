import argparse
import logging
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import tensorflow as tf
import tensorflow_hub as hub
import kipoiseq
from kipoiseq import transforms


class EnformerEmbeddingExtractor:
    def __init__(self, 
                 model_path: str = 'https://tfhub.dev/deepmind/enformer/1',
                 use_gpu: bool = True,
                 mixed_precision: bool = False):
        self.model_path = model_path
        self.sequence_length = 393_216
        self.crop_size = 8192
        self.target_length = 896
        self.num_channels   = 5313

        self._configure_tensorflow(use_gpu, mixed_precision)
        self.model = None
        self._load_model()
        self.transform = self._get_transform()

    def _configure_tensorflow(self, use_gpu: bool, mixed_precision: bool):
        if use_gpu:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logging.info(f"Found {len(gpus)} GPU(s)")
                except RuntimeError as e:
                    logging.warning(f"GPU configuration failed: {e}")
        if mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logging.info("Mixed precision enabled")

    def _load_model(self):
        logging.info(f"Loading Enformer model from {self.model_path}")
        self.model = hub.load(self.model_path).model
        logging.info("Model loaded successfully")

    def _get_transform(self):
        return transforms.Compose([transforms.functional.one_hot_dna])

    def _validate_sequence(self, seq: str) -> str:
        s = seq.upper().replace('U', 'T')
        valid = set('ATCGN')
        if not set(s).issubset(valid):
            bad = set(s) - valid
            raise ValueError(f"Invalid nucleotides: {bad}")
        return s

    def _pad_or_crop_sequence(self, seq: str) -> str:
        n = self.sequence_length
        if len(seq) > n:
            start = (len(seq) - n)//2
            return seq[start:start+n]
        if len(seq) < n:
            pad = n - len(seq)
            lp = pad//2; rp = pad - lp
            return 'N'*lp + seq + 'N'*rp
        return seq

    def _sequence_to_tensor(self, seq: str) -> tf.Tensor:
        s = self._validate_sequence(seq)
        s = self._pad_or_crop_sequence(s)
        one_hot = self.transform(s).astype(np.float32)
        return tf.expand_dims(one_hot, axis=0)

    def extract_embeddings(self, 
                          sequences, 
                          batch_size: int = 1, 
                          return_raw: bool = False):
        all_emb, human_preds, mouse_preds = [], [], []
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting"):
            batch = sequences[i:i+batch_size]
            tensors = [self._sequence_to_tensor(s) for s in batch]
            x = tensors[0] if len(tensors)==1 else tf.concat(tensors, axis=0)
            preds = self.model.predict_on_batch(x)
            if return_raw:
                human_preds.append(preds['human'].numpy())
                mouse_preds.append(preds['mouse'].numpy())
            else:
                h = preds['human'].numpy()
                emb = h.mean(axis=(1,2))
                all_emb.append(emb)
            logging.info(f"Batch {i//batch_size+1}/{(len(sequences)+batch_size-1)//batch_size} done")
        if return_raw:
            return {'human': np.concatenate(human_preds), 'mouse': np.concatenate(mouse_preds)}
        return np.concatenate(all_emb)

    def save_embeddings(self, emb: np.ndarray, path: str, sequences=None):
        p = Path(path)
        if p.suffix == '.npz':
            data = {'embeddings': emb}
            if sequences: data['sequences'] = sequences
            np.savez_compressed(p, **data)
        elif p.suffix == '.npy':
            np.save(p, emb)
        else:
            raise ValueError(f"Unsupported format: {p.suffix}")
        logging.info(f"Saved embeddings to {p}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Enformer embedding + logreg")
    parser.add_argument('-b','--batch_size', type=int, default=10,
                        help='Batch size (default=10)')
    parser.add_argument('-d','--device', choices=['cpu','cuda'], default='cuda',
                        help='Device: cpu or cuda (default=cuda)')
    parser.add_argument('-o','--output_dir', type=str, default='.',
                        help='Path to save outputs (default=.)')
    parser.add_argument('--log', action='store_true',
                        help='Enable logging (INFO)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.log else logging.WARNING,
                        format='[%(asctime)s] %(levelname)s: %(message)s')
    logger = logging.getLogger()

    use_gpu = args.device == 'cuda'
    logger.info(f"Device: {args.device}")

    logger.info(f"Loading dataset...")

    ds = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks")
    train_ds, test_ds = ds['train'], ds['test']

    logger.info(f"Dataset is loaded.")

    logger.info(f"Embedder initialization...")

    extractor = EnformerEmbeddingExtractor(
        use_gpu=use_gpu
    )

    logger.info(f"Embedder is ready!")

    PARAMS_LOGREG = {'max_iter': 1000, 'random_state': 42}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Work starts with the full data....")

    # --- Baseline ---
    baseline = {}
    for task in tqdm(set(train_ds['task']), desc='Baseline'):
        if task in ['splice_sites_all', 'H3K4me2', "promoter_all"]:
            logger.info(f"{task} has been processed. Skip it.")
            continue
        tr = train_ds.filter(lambda x, t=task: x['task']==t)
        te = test_ds.filter(lambda x, t=task: x['task']==t)
        X_tr = extractor.extract_embeddings(tr['sequence'], batch_size=args.batch_size)
        X_te = extractor.extract_embeddings(te['sequence'], batch_size=args.batch_size)
        clf = LogisticRegression(**PARAMS_LOGREG)
        clf.fit(X_tr.reshape(len(X_tr), -1), np.array(tr['label']))
        preds = clf.predict(X_te.reshape(len(X_te), -1))
        baseline[task] = {
            'accuracy': float(accuracy_score(te['label'], preds)),
            'f1_score': float(f1_score(te['label'], preds, average='macro'))
        }
        with open(out_dir / f"results_enformer_task-{task}_baseline.json", 'w') as f:
            json.dump(baseline, f, indent=4)

    logger.info(f"The work with full data has been successfully completed!")
    logger.info(f"Few-shot starts...")

    # --- Few-shot ---
    def few_shot(train, test, ks=(1,5,10,20), trials=5):
        res = {}
        rng = np.random.RandomState(42)
        for task in tqdm(set(train['task']), desc='Few-shot'):
            tr = train.filter(lambda x, t=task: x['task']==t)
            te = test.filter(lambda x, t=task: x['task']==t)
            X_te = extractor.extract_embeddings(te['sequence'], batch_size=args.batch_size)
            y_tr = np.array(tr['label']); y_te = np.array(te['label'])
            res[task] = {}
            for k in ks:
                accs, f1s = [], []
                for _ in range(trials):
                    idxs = []
                    for lbl in np.unique(y_tr):
                        locs = np.where(y_tr==lbl)[0]
                        choice = rng.choice(locs, size=min(k,len(locs)), replace=False)
                        idxs.extend(choice.tolist())
                    X_k = extractor.extract_embeddings([tr['sequence'][i] for i in idxs],
                                                      batch_size=args.batch_size)
                    clf = LogisticRegression(**PARAMS_LOGREG)
                    clf.fit(X_k.reshape(len(X_k), -1), y_tr[idxs])
                    p = clf.predict(X_te.reshape(len(X_te), -1))
                    accs.append(accuracy_score(y_te, p))
                    f1s.append(f1_score(y_te, p, average='macro'))
                res[task][k] = {
                    'accuracy': float(np.mean(accs)),
                    'f1_score': float(np.mean(f1s))
                }

                logger.info(f"Saving results (task={task}, k={k}) in JSON format.")

                with open(out_dir / f"results_enformer_task-{task}_k-{k}.json", 'w') as f:
                    json.dump(res, f, indent=4)
        return res

    results_kshot = few_shot(train_ds, test_ds)

    logger.info(f"Few-shot has been successfully completed!")
    logger.info(f"Saving full results in JSON format...")

    summary = {'full': baseline, 'kshot': results_kshot, 'params': PARAMS_LOGREG}
    with open(out_dir / "results_enformer.json", 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info("All done.")

if __name__ == "__main__":
    main()
