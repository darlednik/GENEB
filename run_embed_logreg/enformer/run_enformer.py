import argparse
import logging
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from tqdm import tqdm
from enformer_pytorch import Enformer
from typing import List


class EnformerPyTorchExtractor:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        # load pretrained Enformer v1 hyperparams
        self.model = Enformer.from_hparams(
            dim=1536,
            depth=11,
            heads=8,
            output_heads=dict(human=5313, mouse=1643),
            target_length=896
        ).to(self.device)
        self.model.eval()
        self.seq_length = 393216

    @staticmethod
    def one_hot_seq(seq: str, seq_length: int):
        """
        Static helper to convert a sequence to one-hot tensor.

        Args:
            seq: Input nucleotide sequence.
            seq_length: Fixed length for cropping/padding and encoding.

        Returns:
            One-hot encoded tensor of shape (4, seq_length).
        """
        mapping = {'A':0, 'C':1, 'G':2, 'T':3}
        # uppercase, replace U->T
        seq = seq.upper().replace('U', 'T')
        # center crop or pad
        L = len(seq)
        if L >= seq_length:
            start = (L - seq_length)//2
            sub = seq[start:start+seq_length]
        else:
            pad = seq_length - L
            left = pad//2
            sub = 'N'*left + seq + 'N'*(pad-left)
        # one-hot
        arr = np.zeros((4, seq_length), dtype=np.float32)
        for i, c in enumerate(sub):
            if c in mapping:
                arr[mapping[c], i] = 1.0
        return torch.tensor(arr)

    def extract_embeddings(self, sequences: List[str], batch_size=1):
        """
        Compute mean-pooled embeddings for a list of genomic sequences.

        Each sequence is one-hot encoded, passed through the Enformer model,
        and the 'human' output track is averaged over spatial and channel dimensions.

        Inputs:
            sequences: List of nucleotide sequences (strings).
            batch_size: Number of sequences to process at once (mem-efficient).

        Returns:
            NumPy array of shape (len(sequences),) containing one embedding per sequence.
        """
        embs = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                # prepare batch tensor (b,4,L)
                tensor = torch.stack([self.one_hot_seq(s, self.seq_length) for s in batch], dim=0).permute(0, 2, 1)
 
                tensor = tensor.to(self.device)
                out = self.model(tensor)
                human = out['human']  # shape (b,896,5313)
                # mean pool spatial dims and channels -> (b,)
                emb = human.mean(dim=(1, 2)).cpu().numpy()
                embs.append(emb)
        return np.concatenate(embs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-o', '--output_dir', type=str, default='.')
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.log else logging.WARNING,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)

    device = args.device
    batch_size = args.batch_size
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f'Initializing Enformer extractor on {device}')
    extractor = EnformerPyTorchExtractor(device=device)

    logger.info('Loading dataset')
    ds = load_dataset('InstaDeepAI/nucleotide_transformer_downstream_tasks')
    train_ds, test_ds = ds['train'], ds['test']

    PARAMS_LOGREG = {'max_iter': 1000, 'random_state': 42}
    baseline = {}

    logger.info('Running baseline (full-data)')
    for task in tqdm(set(train_ds['task']), desc='Baseline'):
        tr = train_ds.filter(lambda x, t=task: x['task'] == t)
        te = test_ds.filter(lambda x, t=task: x['task'] == t)
        seqs_tr, y_tr = tr['sequence'], np.array(tr['label'])
        seqs_te, y_te = te['sequence'], np.array(te['label'])

        X_tr = extractor.extract_embeddings(seqs_tr, batch_size)
        X_te = extractor.extract_embeddings(seqs_te, batch_size)

        clf = LogisticRegression(**PARAMS_LOGREG)
        clf.fit(X_tr.reshape(len(X_tr), -1), y_tr)
        p = clf.predict(X_te.reshape(len(X_te), -1))

        baseline[task] = {
            'accuracy': float(accuracy_score(y_te, p)),
            'f1_score': float(f1_score(y_te, p, average='macro'))
        }
        with open(out_dir / f'result_enformer_task-{task}_baseline.json', 'w') as f:
            json.dump(baseline[task], f, indent=4)

    logger.info('Running few-shot')
    def few_shot(train, test, ks=(1,5,10,20), trials=5):
        res = {}
        rng = np.random.RandomState(42)
        for task in tqdm(set(train['task']), desc='Few-shot'):
            tr = train.filter(lambda x, t=task: x['task'] == t)
            te = test.filter(lambda x, t=task: x['task'] == t)
            seqs_te, y_te = te['sequence'], np.array(te['label'])
            X_te = extractor.extract_embeddings(seqs_te, batch_size)
            y_tr = np.array(tr['label'])
            res[task] = {}
            for k in ks:
                accs, f1s = [], []
                for _ in range(trials):
                    idxs = []
                    for lbl in np.unique(y_tr):
                        locs = np.where(y_tr == lbl)[0]
                        choice = rng.choice(locs, size=min(k, len(locs)), replace=False)
                        idxs.extend(choice.tolist())
                    seqs_k = [tr['sequence'][i] for i in idxs]
                    y_k = y_tr[idxs]
                    X_k = extractor.extract_embeddings(seqs_k, batch_size)
                    clf = LogisticRegression(**PARAMS_LOGREG)
                    clf.fit(X_k.reshape(len(X_k), -1), y_k)
                    p = clf.predict(X_te.reshape(len(X_te), -1))
                    accs.append(accuracy_score(y_te, p))
                    f1s.append(f1_score(y_te, p, average='macro'))
                res[task][k] = {
                    'accuracy': float(np.mean(accs)),
                    'f1_score': float(np.mean(f1s))
                }
                with open(out_dir / f'result_enformer_{task}_k{k}.json', 'w') as f:
                    json.dump(res[task][k], f, indent=4)
        return res

    results_kshot = few_shot(train_ds, test_ds)

    summary = {'full': baseline, 'kshot': results_kshot}
    with open(out_dir / 'results_enformer.json', 'w') as f:
        json.dump(summary, f, indent=4)

    logger.info('Done')

if __name__ == "__main__":
    main()