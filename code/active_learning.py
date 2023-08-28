from enum import Enum

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

from config import classificator_names, get_classificator, active_learning_methods, get_active_learning_method, ActiveLearner, Committee
from preprocessing import standardize, split_dataset_xy
from metrics import method_eval

# Uncertainty sampling
import numpy as np
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
def uncertainty_sampling_al(**kwargs):
    results = {"pooled_training":[]}
    
    y_train_init = kwargs["y_train_init"]
    # initializing the learner
    learner = ActiveLearner(
        estimator=kwargs["Clf"](**kwargs["clf_params"]),
        X_training=kwargs["X_train_init"].to_numpy(), y_training=y_train_init.values,
        query_strategy = kwargs["alm_params"]["query_strategy"]
    )

    X_pool = kwargs["X_pool"].to_numpy()
    y_pool = kwargs["y_pool"].values
    X_test = kwargs["X_test"].to_numpy()
    y_test = kwargs["y_test"].values

    classes_distribution = y_train_init.value_counts()
    results["train_init"] = {"classes": classes_distribution.index.tolist(),
                             "counts": classes_distribution.values.tolist()}

    y_pred = learner.predict(X_test)
    metrics = method_eval(y_test, y_pred)
    results["pooled_training"].append(metrics)

    for index in range(kwargs["alm_params"]["max_iterations"]):
        query_index, query_instance = learner.query(X_pool)

        # Teach our ActiveLearner model the record it has requested.
        X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
        learner.teach(X=X, y=y)

        # Remove the queried instance from the unlabeled pool.
        X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

        y_pred = learner.predict(X_test)
        metrics = method_eval(y_test, y_pred)
        results["pooled_training"].append(metrics)
        
    return results

# QueryByCommittee
import numpy as np
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
def query_by_committee_al(**kwargs):
    n_members = kwargs["alm_params"]["n_models"]
    learners = []
    results = {"pooled_training":[]}

    y_train_init = kwargs["y_train_init"]
    for _ in range(n_members):
        learner = ActiveLearner(
                estimator=kwargs["Clf"](**kwargs["clf_params"]),
                X_training=kwargs["X_train_init"].to_numpy(), y_training=y_train_init.values
            )
        learners.append(learner)

    committee = Committee(
        learner_list=learners,
        query_strategy = kwargs["alm_params"]["disagreement_measure"]
    )

    X_pool = kwargs["X_pool"].to_numpy()
    y_pool = kwargs["y_pool"].values
    X_test = kwargs["X_test"].to_numpy()
    y_test = kwargs["y_test"].values

    classes_distribution = y_train_init.value_counts()
    results["train_init"] = {"classes": classes_distribution.index.tolist(),
                             "counts": classes_distribution.values.tolist()}

    y_pred = committee.predict(X_test)
    metrics = method_eval(y_test, y_pred)
    results["pooled_training"].append(metrics)

    for index in range(kwargs["alm_params"]["max_iterations"]):
        query_idx, query_instance = committee.query(X_pool)

        # Teach our Committee model the record it has requested.
        X, y = X_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(1, )
        committee.teach(X=X, y=y)

        # Remove the queried instance from the unlabeled pool.
        X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx)

        y_pred = committee.predict(X_test)
        metrics = method_eval(y_test, y_pred)
        results["pooled_training"].append(metrics)

    return results

    # # for learner in committee:
    # #     # ...do something with the learner...
    # samp_idx = np.random.choice(len(X_pool))
    # print(committee.vote(X_pool[samp_idx].reshape(1,-1)))
    # print(committee.vote_proba(X_pool[samp_idx].reshape(1,-1)))

    # # To get the predictions and class probabilities of the Committee itself, you shall use the .predict(X) and .predict_proba() methods

# Classification methods switcher
def learn_active(**kwargs) -> list[int]:
    if kwargs["method_name"] == 'uncertainty_sampling':
        al_results = uncertainty_sampling_al(**kwargs)
    elif kwargs["method_name"] == 'query_by_committee':
        al_results = query_by_committee_al(**kwargs)

    return al_results

# Main loop
RANDOM_STATE_SEED = 13

class DebugLevel(Enum):
    BENCHMARK = 0
    EACH_DATASET = 1
    ACTIVE_LEARNING = 2

def test_al_methods(datasets, debug_level = 0):
    results = []
    # Classificator models
    for classificator_name in classificator_names:
        clf_info = get_classificator(classificator_name)
        Clf = clf_info.get("model")
        clf_params_arr = clf_info.get("params")

        # Classificator model parameters
        for clf_params in clf_params_arr:
            
            # Active learning methods
            for method_name in active_learning_methods:
                alm_info = get_active_learning_method(method_name)
                alm_params_arr = alm_info.get("params")

                # Active learning method parameters
                for alm_params in alm_params_arr:

                    results.append({"classificator": str(Clf.__name__),
                                    "classificator_params": clf_params,
                                    "al_method": str(method_name),
                                    "al_method_params": alm_params,
                                    "datasets_results": []
                                    })

                    all_datasets_precision_arr = []
                    all_datasets_recall_arr = []
                    all_datasets_f1_score_arr = []
                    all_datasets_auc_pr_curve_arr = [] 
                    all_datasets_auc_roc_curve_arr = []
                    all_datasets_g_mean = []

                    # Datasets
                    for dataset_name, dataset in datasets.items():

                        # Standardizing
                        dataset = standardize(dataset)

                        # Spliting
                        X, y = split_dataset_xy(dataset)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE_SEED)
                    
                        fold_results_arr = []
                        al_results_arr = []

                        # KFold
                        skf = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE_SEED, shuffle=True)
                        for kf_train_indices, kf_test_indices in skf.split(X_train, y_train):

                            # Kfold train and test
                            X_train_kf, X_test_kf, y_train_kf, y_test_kf = X_train.iloc[kf_train_indices], \
                            X_train.iloc[kf_test_indices], y_train.iloc[kf_train_indices], y_train.iloc[kf_test_indices]
                            
                            # Kfold pool form train
                            X_train_init, X_pool, y_train_init, y_pool = train_test_split(X_train_kf, y_train_kf, \
                                                                            test_size=alm_params.get("pool_size"), stratify=y_train_kf, random_state=RANDOM_STATE_SEED)
                            
                            al_results = \
                                learn_active(Clf=Clf, clf_params=clf_params, method_name=method_name, alm_params=alm_params, X_train_init=X_train_init, X_pool=X_pool, \
                                X_test=X_test_kf, y_train_init=y_train_init, y_pool=y_pool, y_test=y_test_kf)
                            
                            fold_results_arr.append(al_results["pooled_training"][-1])
                            al_results_arr.append(al_results)

                        print(50*"-"+"\n",\
                            f"Classificator: {results[-1]['classificator']}\n",\
                            f"Classificator parameters: {results[-1]['classificator_params']}\n",\
                            f"Active learning method: {results[-1]['al_method']}\n",\
                            f"Active leraning parameters: {results[-1]['al_method_params']}\n",\
                            f"Dataset: {dataset_name}\n",\
                            50*"-"+"\n", sep='', end='\n')

                        # Result for benchmark dataset - mean of kfolds
                        mean_precision = np.mean([x["precision"] for x in fold_results_arr])
                        mean_recall = np.mean([x["recall"] for x in fold_results_arr])
                        mean_f1_score = np.mean([x["f1_score"] for x in fold_results_arr])
                        mean_auc_pr_curve = np.mean([x["auc_pr_curve"] for x in fold_results_arr])
                        mean_auc_roc_curve = np.mean([x["auc_roc_curve"] for x in fold_results_arr])
                        mean_g_mean = np.mean([x["g_mean"] for x in fold_results_arr])

                        print(f"Precision: {mean_precision:.2f}\n",\
                            f"Recall: {mean_recall:.2f}\n",\
                            f"F1 score: {mean_f1_score:.2f}\n",\
                            f"Precision-Recall AUC: {mean_auc_pr_curve:.2f}\n",\
                            f"ROC AUC: {mean_auc_roc_curve:.2f}\n",\
                            f"G mean: {mean_g_mean:.2f}\n", sep='', end='\n')
                        
                        results[-1]["datasets_results"].append({"dataset_name": dataset_name, 
                                                                "al_results": al_results_arr, # should it be averaged?
                                                                "fold_results": fold_results_arr, 
                                                                "results": {
                                                                        "precision": mean_precision,
                                                                        "recall": mean_recall,
                                                                        "f1_score": mean_f1_score,
                                                                        "auc_pr_curve": mean_auc_pr_curve,
                                                                        "auc_roc_curve": mean_auc_roc_curve,
                                                                        "g_mean": mean_g_mean
                                                                        }
                                                                })
                        
                        all_datasets_precision_arr.append(mean_precision)
                        all_datasets_recall_arr.append(mean_recall)
                        all_datasets_f1_score_arr.append(mean_f1_score)
                        all_datasets_auc_pr_curve_arr.append(mean_auc_pr_curve)
                        all_datasets_auc_roc_curve_arr.append(mean_auc_roc_curve)
                        all_datasets_g_mean.append(mean_g_mean)
                    
                    # Result for all benchmark datasets - mean of all benchmark datasets
                    mean_precision = np.mean(all_datasets_precision_arr)
                    mean_recall = np.mean(all_datasets_recall_arr)
                    mean_f1_score = np.mean(all_datasets_f1_score_arr)
                    mean_auc_pr_curve = np.mean(all_datasets_auc_pr_curve_arr)
                    mean_auc_roc_curve = np.mean(all_datasets_auc_roc_curve_arr)
                    mean_g_mean = np.mean(all_datasets_g_mean)

                    results[-1]["benchmark_result"] = {
                        "precision": mean_precision,
                        "recall": mean_recall,
                        "f1_score": mean_f1_score,
                        "auc_pr_curve": mean_auc_pr_curve,
                        "auc_roc_curve": mean_auc_roc_curve,
                        "g_mean": mean_g_mean
                    }
                    
                    print(50*"-"+"\n",\
                            f"Classificator: {Clf.__name__}\n",\
                            f"Classificator parameters: {clf_params}\n",\
                            f"Active learning method: {method_name}\n",\
                            f"Active leraning parameters: {alm_params}\n",\
                            f"All datasets\n",\
                            50*"-"+"\n", sep='', end='\n')
                    
                    print(f"Precision: {mean_precision:.2f}\n",\
                        f"Recall: {mean_recall:.2f}\n",\
                        f"F1 score: {mean_f1_score:.2f}\n",\
                        f"Precision-Recall AUC: {mean_auc_pr_curve:.2f}\n",\
                        f"ROC AUC: {mean_auc_roc_curve:.2f}\n",\
                        f"G mean: {mean_g_mean:.2f}\n", sep='', end='\n')
                    
    return results