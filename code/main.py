from benchmark import Benchmark

datasets = Benchmark.load_all_benchmark_datasets()

datasets = datasets["ecoli"]

from active_learning import test_al_methods

results = test_al_methods(datasets)

with open("results.json", "w") as outfile:
        outfile.write(str(results).replace("\'", '\"'))