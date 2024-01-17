from benchmark import Benchmark
from config import DATASETS

datasets = Benchmark.load_all_benchmark_datasets()

# Select datasets
datasets = {key: datasets[key] for key in DATASETS}

from active_learning import test_al_methods

results = test_al_methods(datasets)

with open("results.json", "w") as outfile: # TODO: could be changed to json dump
        outfile.write(str(results).replace("\'", '\"'))