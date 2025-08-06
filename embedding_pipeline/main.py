import argparse
import logging
import json
import importlib
import pandas as pd
from pathlib import Path
from pipeline.classification import EmbeddingClassificationPipeline


def load_extractor_class(module_name: str, class_name: str):
    module = importlib.import_module(f"extractors.{module_name}")
    return getattr(module, class_name)


def load_csv_dataset(file_path):
    df = pd.read_csv(file_path)
    required_cols = {"text", "label", "split"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns in {file_path}: {required_cols - set(df.columns)}")
    return df.to_dict(orient="records")


def main(**kwargs):
    csv_dir = Path(kwargs.get("csv_dir"))
    output_dir = kwargs.get("output_dir")
    extractor_name = kwargs.get("extractor")
    module_name = kwargs.get("module") or extractor_name.lower()
    device = kwargs.get("device")
    batch_size = kwargs.get("batch_size")
    name_model = kwargs.get("name_model")

    import torch
    logging.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    ExtractorClass = load_extractor_class(module_name, extractor_name)
    extractor = ExtractorClass(device=device, name_model=name_model)

    pipeline = EmbeddingClassificationPipeline(
        extractor,
        output_directory=output_dir,
        batch_size=batch_size
    )

    for csv_path in sorted(csv_dir.glob("*.csv")):
        task_name = csv_path.stem
        logging.info(f"Processing task: {task_name}")
        dataset = load_csv_dataset(csv_path)
        pipeline.evaluate_from_csv(dataset, task_name=task_name)

    logging.info("All tasks completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_dir", type=str, required=True,
        help="Path to directory with .csv files"
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
        "--name_model", type=str, default=None,
        help="Path or name of the model to load"
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s"
    )

    main(**vars(args))