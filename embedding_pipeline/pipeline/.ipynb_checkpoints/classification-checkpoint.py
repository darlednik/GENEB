import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from collections import defaultdict

class EmbeddingClassificationPipeline:
    def __init__(self, extractor, output_directory: str, batch_size: int = 10):
        self.extractor = extractor
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size
        self.logreg_params = {'max_iter': 1000, 'random_state': 42}

    def evaluate_from_csv(self, dataset: list[dict], task_name: str, shots=(1, 5, 10, 20), trials=5):
        train = [x for x in dataset if x['split'] == 'train']
        test = [x for x in dataset if x['split'] == 'test']

        train_sequences = [x['text'] for x in train]
        train_labels = np.array([x['label'] for x in train])
        test_sequences = [x['text'] for x in test]
        test_labels = np.array([x['label'] for x in test])

        train_embeddings = self.extractor.extract_embeddings(train_sequences, self.batch_size)
        test_embeddings = self.extractor.extract_embeddings(test_sequences, self.batch_size)

        extractor_name = self.extractor.__class__.__name__.lower()

        full_metrics = defaultdict(list)
        for _ in range(trials):
            clf = LogisticRegression(**self.logreg_params).fit(train_embeddings, train_labels)
            preds = clf.predict(test_embeddings)
            full_metrics['accuracy'].append(accuracy_score(test_labels, preds))
            full_metrics['f1_score'].append(f1_score(test_labels, preds, average='macro'))
            full_metrics['mcc'].append(matthews_corrcoef(test_labels, preds))

        full_results = {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
            for metric, values in full_metrics.items()
        }
        self._save_result(task_name, extractor_name, full_results)

        few_shot_results = {}
        rng = np.random.RandomState(42)

        for k in shots:
            accs, f1s, mccs = [], [], []
            for _ in range(trials):
                idxs = np.concatenate([
                    locs if k >= len(locs) else rng.choice(locs, size=k, replace=False)
                    for cls in np.unique(train_labels)
                    for locs in [np.where(train_labels == cls)[0]]
                ])

                clf = LogisticRegression(**self.logreg_params).fit(train_embeddings[idxs], train_labels[idxs])
                preds = clf.predict(test_embeddings)

                accs.append(accuracy_score(test_labels, preds))
                f1s.append(f1_score(test_labels, preds, average='macro'))
                mccs.append(matthews_corrcoef(test_labels, preds))

            few_shot_results[k] = {
                'accuracy': {'mean': float(np.mean(accs)), 'std': float(np.std(accs))},
                'f1_score': {'mean': float(np.mean(f1s)), 'std': float(np.std(f1s))},
                'mcc': {'mean': float(np.mean(mccs)), 'std': float(np.std(mccs))}
            }

            self._save_result(task_name, f"{extractor_name}_k-{k}", few_shot_results[k])

        return {
            "full": full_results,
            "few_shot": few_shot_results
        }

    def _save_result(self, task_name: str, result_type: str, result_data):
        path = self.output_directory / f"results_{task_name}_{result_type}.json"
        with open(path, 'w') as f:
            json.dump(result_data, f, indent=4)
