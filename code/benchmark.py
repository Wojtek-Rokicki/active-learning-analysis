import os

import sklearn
import pandas as pd
from imblearn.datasets import fetch_datasets

class Benchmark:
    # Benchmark datasets
    # Source: https://imbalanced-learn.org/stable/datasets/index.html
    datasets_names = [      # ID    Repository & Target             Ratio     #S      #F
        "ecoli",            # 1     UCI, target: imU                8.6:1     336     7
        "optical_digits",   # 2     UCI, target: 8                  9.1:1     5,620   64
        "satimage",         # 3     UCI, target: 4                  9.3:1     6,435   36
        "pen_digits",       # 4     UCI, target: 5                  9.4:1     10,992  16
        "abalone",          # 5     UCI, target: 7                  9.7:1     4,177   10
        "sick_euthyroid",   # 6     UCI, target: sick euthyroid     9.8:1     3,163   42
        "spectrometer",     # 7     UCI, target: >=44               11:1      531     93
        "car_eval_34",      # 8     UCI, target: good, v good       12:1      1,728   21
        "isolet",           # 9     UCI, target: A, B               12:1      7,797   617
        "us_crime",         # 10    UCI, target: >POOL_SIZE5        12:1      1,994   100
        "yeast_ml8",        # 11    LIBSVM, target: 8               13:1      2,417   103
        "scene",            # 12    LIBSVM, target: >one label      13:1      2,407   294
        "libras_move",      # 13    UCI, target: 1                  14:1      360     90
        "thyroid_sick",     # 14    UCI, target: sick               15:1      3,772   52
        "coil_2000",        # 15    KDD, CoIL, target: minority     16:1      9,822   85
        "arrhythmia",       # 16    UCI, target: 06                 17:1      452     278
        "solar_flare_m0",   # 17    UCI, target: M->0               19:1      1,389   32
        "oil",              # 18    UCI, target: minority           22:1      937     49
        "car_eval_4",       # 19    UCI, target: vgood              26:1      1,728   21
        "wine_quality",     # 20    UCI, wine, target: <=4          26:1      4,898   11
        "letter_img",       # 21    UCI, target: Z                  26:1      20,000  16
        "yeast_me2",        # 22    UCI, target: ME2                28:1      1,484   8
        "webpage",          # 23    LIBSVM, w7a, target: minority   33:1      34,780  300
        "ozone_level",      # 24    UCI, ozone, data                34:1      2,536   72
        "mammography",      # 25    UCI, target: minority           42:1      11,183  6
        "protein_homo",     # 26    KDD CUP 2004, minority          11:1      145,751 74
        "abalone_19"        # 27    UCI, target: 19                 130:1     4,177   10
    ]               
    #   "htru2"             # fin   UCI, target: minority           10:1      17,898  8

    
    cwd = os.getcwd()
    fp_benchmark = os.path.join(cwd, "../data/benchmark")

    # Download, convert and save datasets
    def save_benchmark_dataset(bunch: sklearn.utils.Bunch) -> None:
        df = pd.DataFrame(bunch['data'])
        df['target'] = bunch['target']
        df.name = bunch['DESCR']

        df.to_csv(os.path.join(Benchmark.fp_benchmark, f'{df.name}.csv'))

    def save_all_benchmark_datasets():
        for dataset_name in Benchmark.datasets_names:
            bunch = fetch_datasets()[dataset_name]
            Benchmark.save_benchmark_dataset(bunch)

    # Load datasets
    def load_benchmark_dataset(name: str) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(Benchmark.fp_benchmark, f'{name}.csv'), index_col=0)
        df.name = name
        return df

    def load_all_benchmark_datasets():
        datasets = {}
        for dataset_name in Benchmark.datasets_names:
            df = Benchmark.load_benchmark_dataset(dataset_name)
            datasets[dataset_name] = df
        return datasets