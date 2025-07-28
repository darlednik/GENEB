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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True, help="Path to directory with .csv files")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for results")
    parser.add_argument("--extractor", type=str, required=True, help="Extractor class name")
    parser.add_argument("--module", type=str, default=None, help="Optional module name. Default: lowercase of class name.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    module_name = args.module or args.extractor.lower()
    ExtractorClass = load_extractor_class(module_name, args.extractor)
    extractor = ExtractorClass(device=args.device)

    pipeline = EmbeddingClassificationPipeline(extractor, output_directory=args.output_dir, batch_size=args.batch_size)

    csv_dir = Path(args.csv_dir)
    for csv_path in sorted(csv_dir.glob("*.csv")):
        task_name = csv_path.stem
        logging.info(f"Processing task: {task_name}")
        dataset = load_csv_dataset(csv_path)
        pipeline.evaluate_from_csv(dataset, task_name=task_name)

    logging.info("All tasks completed")


if __name__ == "__main__":
    main()