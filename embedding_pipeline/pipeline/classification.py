import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from collections import defaultdict
import logging
from typing import List, Dict, Tuple

class EmbeddingClassificationPipeline:
    def __init__(self, extractor, name_model, output_directory: str, batch_size: int = 10, n_jobs: int = 32):
        self.extractor = extractor
        self.name_model = name_model
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.logreg_params = {'max_iter': 1000, 'n_jobs': self.n_jobs}
        
        logging.info(f"Logreg params: {self.logreg_params}")

    @staticmethod
    def _prepare_data_from_csv(dataset: List[Dict], split: str) -> Tuple[List[str], np.array]:
        
        data_split = [x for x in dataset if x['split'] == split]

        data_split_sequences = [x['text'] for x in data_split]
        data_split_labels = np.array([x['label'] for x in data_split])
        
        return data_split_sequences, data_split_labels

    @staticmethod
    def _prepare_data_dnalongbench(dataset: List[Dict], split: str, task_name: str) -> Tuple[List[str], np.array]:
        
        data_split = [x for x in dataset if x['split'] == split]

        if task_name.split("@@")[0] == "eqtl_prediction":
            data_split = [x for x in dataset if x['split'] == split]

            data_split_labels = np.array([x['y'][0] for x in data_split])
        
            return data_split, data_split_labels

        elif task_name.split("@@")[0] == "enhancer_target_gene_prediction":
            
            data_split = [x for x in dataset if x['split'] == split]
            
            data_split_sequences = [x['sequence'][0] for x in data_split]
            data_split_labels = np.array([x['label'][0] for x in data_split])

            return data_split_sequences, data_split_labels
        
        else:
            raise ValueError(f"Missing {task_name.split('@@')[0]}. Available task_names: eqtl_prediction, enhancer_target_gene_prediction")
    
    def evaluate(self, dataset: list[dict], task_name: str, format_reader: str, shots=(1, 10), seeds=(13, 17, 42, 123, 997)) -> dict:
        
        if format_reader == "dnalongbench":
            if task_name.split("@@")[0] == 'eqtl_prediction':
                train_sequences, train_labels = self._prepare_data_dnalongbench(dataset, split="train", task_name=task_name)
                test_sequences, test_labels = self._prepare_data_dnalongbench(dataset, split="test", task_name=task_name)
            
                train_embeddings_x_ref = self.extractor.extract_embeddings([x['x_ref'][0] for x in train_sequences], self.batch_size)
                test_embeddings_x_ref = self.extractor.extract_embeddings([x['x_ref'][0] for x in test_sequences], self.batch_size)

                train_embeddings_x_alt = self.extractor.extract_embeddings([x['x_alt'][0] for x in train_sequences], self.batch_size)
                test_embeddings_x_alt = self.extractor.extract_embeddings([x['x_alt'][0] for x in test_sequences], self.batch_size)

                train_embeddings = train_embeddings_x_alt - train_embeddings_x_ref
                test_embeddings = test_embeddings_x_alt - test_embeddings_x_ref
                

            elif task_name.split("@@")[0] == "enhancer_target_gene_prediction":
                train_sequences, train_labels = self._prepare_data_dnalongbench(dataset, split="train", task_name=task_name)
                test_sequences, test_labels = self._prepare_data_dnalongbench(dataset, split="test", task_name=task_name)

                train_embeddings = self.extractor.extract_embeddings(train_sequences, self.batch_size)
                test_embeddings = self.extractor.extract_embeddings(test_sequences, self.batch_size)


        elif format_reader == "csv":

            train_sequences, train_labels = self._prepare_data_from_csv(dataset, split="train")
            test_sequences, test_labels = self._prepare_data_from_csv(dataset, split="test")

            train_embeddings = self.extractor.extract_embeddings(train_sequences, self.batch_size)
            test_embeddings = self.extractor.extract_embeddings(test_sequences, self.batch_size)


        extractor_name = self.extractor.__class__.__name__.lower()

        full_metrics = defaultdict(list)
        
        for seed in seeds:
            params = dict(self.logreg_params, random_state=seed)
            clf = LogisticRegression(**params).fit(train_embeddings, train_labels)

            preds = clf.predict(test_embeddings)
            full_metrics['accuracy'].append(accuracy_score(test_labels, preds))
            full_metrics['f1_score'].append(f1_score(test_labels, preds, average='macro'))
            full_metrics['mcc'].append(matthews_corrcoef(test_labels, preds))
            
            if format_reader == 'dnalongbench':
                full_metrics['rocauc'].append(roc_auc_score(test_labels, preds))

        full_results = {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
            for metric, values in full_metrics.items()
        }
        self._save_result(task_name, extractor_name, full_results)

        few_shot_results = {}

        for k in shots:
            accs, f1s, mccs, rocaucs = [], [], [], []
            for seed in seeds:
                rng = np.random.RandomState(seed)

                idxs = np.concatenate([
                    rng.choice(locs, size=min(k, len(locs)), replace=False)
                    for cls in np.unique(train_labels)
                    for locs in [np.where(train_labels == cls)[0]]
                ])
                
                params = dict(self.logreg_params, random_state=seed)
                clf = LogisticRegression(**self.logreg_params).fit(train_embeddings[idxs], train_labels[idxs])
                preds = clf.predict(test_embeddings)

                accs.append(accuracy_score(test_labels, preds))
                f1s.append(f1_score(test_labels, preds, average='macro'))
                mccs.append(matthews_corrcoef(test_labels, preds))

                if format_reader == "dnalongbench":
                    rocaucs.append(roc_auc_score(test_labels, preds))

            few_shot_results[k] = {
                'accuracy': {'mean': float(np.mean(accs)), 'std': float(np.std(accs))},
                'f1_score': {'mean': float(np.mean(f1s)), 'std': float(np.std(f1s))},
                'mcc': {'mean': float(np.mean(mccs)), 'std': float(np.std(mccs))},
            }

            if format_reader == "dnalongbench":
                few_shot_results[k]['aucroc'] = {'mean': float(np.mean(rocaucs)), 'std': float(np.std(rocaucs))}

            self._save_result(task_name, f"{extractor_name}_k-{k}", few_shot_results[k])

        return {
            "full": full_results,
            "few_shot": few_shot_results
        }

    def _save_result(self, task_name: str, result_type: str, result_data):
        path = self.output_directory / f"results_{Path(self.name_model).stem}_{task_name}_{result_type}.json"
        with open(path, 'w') as f:
            json.dump(result_data, f, indent=4)
