import argparse
import logging
import sys
import json
import importlib
import pandas as pd
from pathlib import Path
from pipeline.classification import EmbeddingClassificationPipeline
import torch
from utility_data_modules.eqtl.DNALongBench.dnalongbench.utils import load_data          

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

PROJECT_ROOT = Path(__file__).parent
PATHS = {
    'PanGeneGraphTrans': PROJECT_ROOT / 'utility_modules' / 'deepgene' / 'DeepGene' / 'PanGeneGraphTrans',
    'SPACE': PROJECT_ROOT / "utility_modules" / "SPACE"
}


for name, path in PATHS.items():
    if path.exists():
        sys.path.insert(0, str(path))
        logging.info(f"Added {name}: {path}")


DICT_DNALONGBENCH_NAME_TASK_SUBSET = {"enhancer_target_gene_prediction": [None],
                     "eqtl_prediction": ['Adipose_Subcutaneous', 'Artery_Tibial', 'Cells_Cultured_fibroblasts', 'Muscle_Skeletal', 'Nerve_Tibial', 'Skin_Not_Sun_Exposed_Suprapubic', 'Skin_Sun_Exposed_Lower_leg', 'Thyroid', 'Whole_Blood']
                    }


def load_extractor_class(module_name: str, class_name: str):
    module = importlib.import_module(f"extractors.{module_name}")
    return getattr(module, class_name)


def load_csv_dataset(file_path):
    df = pd.read_csv(file_path)
    required_cols = {"text", "label", "split"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns in {file_path}: {required_cols - set(df.columns)}")
    return df.to_dict(orient="records")

def load_dnalongbench(root: str, task_name: str, subset: str, batch_size: int):
    train, _, test = load_data(
        root=root,
        task_name=task_name,
        subset=subset,
        batch_size=batch_size
    )

    return ([{**kv, "split": "train"} for kv in list(train)] + 
            [{**kv, "split": "test"} for kv in list(test)])

def main(**kwargs):
    data_dir = Path(kwargs.get("data_dir"))
    output_dir = kwargs.get("output_dir")
    extractor_name = kwargs.get("extractor")
    module_name = kwargs.get("module") or extractor_name.lower()
    device = kwargs.get("device")
    batch_size = kwargs.get("batch_size")
    name_model = kwargs.get("name_model")
    n_jobs = kwargs.get("n_jobs")
    format_reader = kwargs.get("format_reader")
    type_train = kwargs.get("type_train", "all")


    logging.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    ExtractorClass = load_extractor_class(module_name, extractor_name)
    extractor = ExtractorClass(device=device, name_model=name_model)

    pipeline = EmbeddingClassificationPipeline(
        extractor,
        name_model,
        output_directory=output_dir,
        batch_size=batch_size,
        n_jobs=n_jobs
    )

    if format_reader == "csv":
        for csv_path in sorted(data_dir.glob("*.csv")):
            task_name = csv_path.stem
            logging.info(f"Processing task: {task_name}")
            dataset = load_csv_dataset(csv_path)
            pipeline.evaluate(dataset, task_name=task_name, format_reader=format_reader, type_train=type_train)

    elif format_reader == "dnalongbench":
        for task_name, list_subsets in DICT_DNALONGBENCH_NAME_TASK_SUBSET.items():
            for subset in list_subsets:
                logging.info(f"Processing DNALongBench, task_name: {task_name}, subset: {subset}")

                dataset = load_dnalongbench(root=data_dir, task_name=task_name, subset=subset, batch_size=1)
                pipeline.evaluate(dataset, task_name=task_name + "@@" + (subset if subset else ''), format_reader=format_reader, type_train=type_train)
    

    logging.info("All tasks completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to directory with .csv files or benchmark folders"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory for results"
    )
    parser.add_argument(
        "--extractor", type=str, required=True,
        help="Extractor class name"
    )
    parser.add_argument(
        "--module", type=str, default=None,
        help="Optional module name. Default: lowercase of class name."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for computation (cpu or cuda)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--name_model", type=str, required=True,
        help="Path or name of the model to load"
    )

    parser.add_argument(
        "--n_jobs", type=int, default=32,
        help="Number jobs for training logreg"
    
    )

    parser.add_argument(
        "--format_reader", type=str, default=None,
        help="Formats: csv, dnalongbench"
    )

    parser.add_argument(
        "--type_train", type=str, default="all",
        help="A type of train process. Is avaiable: ['all', 'only_few_shot', 'only_full']. Default: 'all'"
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s"
    )

    main(**vars(args))
