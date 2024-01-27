import os

import sklearn
import pandas as pd
from imblearn.datasets import fetch_datasets

from config import DATASETS_NAMES, BENCHMARKS_PATH

class Benchmark:
    """
    Class managing benchmark datasets.
    
    References
    ----------
    Source: https://imbalanced-learn.org/stable/datasets/index.html
    
    """
    
    _datasets_names: list = DATASETS_NAMES
    _benchmarks_filepath: str = BENCHMARKS_PATH

    # Convert and save dataset
    def _save_benchmark_dataset(bunch: sklearn.utils.Bunch) -> None:
        df = pd.DataFrame(bunch['data'])
        df['target'] = bunch['target']
        df.name = bunch['DESCR']

        df.to_csv(os.path.join(Benchmark._benchmarks_filepath, f'{df.name}.csv'))

    # Download and save all datasets
    def save_all_benchmark_datasets() -> None:
        for dataset_name in Benchmark._datasets_names:
            bunch = fetch_datasets()[dataset_name]
            Benchmark._save_benchmark_dataset(bunch)

    # Load dataset from benchmark filepath
    def _load_benchmark_dataset(name: str) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(Benchmark._benchmarks_filepath, f'{name}.csv'), index_col=0)
        df.name = name
        return df

    # Load all datasets
    def load_all_benchmark_datasets() -> dict:
        datasets = {}
        for dataset_name in Benchmark._datasets_names:
            df = Benchmark._load_benchmark_dataset(dataset_name)
            datasets[dataset_name] = df
        return datasets