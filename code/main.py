from benchmark import Benchmark

datasets = Benchmark.load_all_benchmark_datasets()

from active_learning import test_al_methods

test_al_methods(datasets)

# TODO: plot results
