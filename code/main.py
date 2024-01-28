from benchmark import Benchmark

datasets = Benchmark.load_all_benchmark_datasets()

from active_learning import test_al_methods

test_al_methods(datasets)

from utils import read_all_n_kcv_agg_results

results = read_all_n_kcv_agg_results()

from plotting import plot_all_n_kcv_agg_all_metrics_results, \
                        plot_all_n_kcv_agg_one_metric_results

plot_all_n_kcv_agg_all_metrics_results(results)

plot_all_n_kcv_agg_one_metric_results(results, "auc_pr_curve")
