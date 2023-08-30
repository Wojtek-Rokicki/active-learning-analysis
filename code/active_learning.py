from enum import Enum
from collections import Counter

import os, time, json

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from config import *
from preprocessing import standardize, split_dataset_xy
from model_selection import tune_classifiers_hyperparameters
from metrics import method_eval, f2_score
from utils import save_model
from plotting import plot_pr_curve, plot_roc

from modAL import ActiveLearner, Committee
from modAL.uncertainty import classifier_entropy

try:
    os.makedirs(PARTIAL_RESULTS_PATH)
except FileExistsError:
    pass

# Active Learning querying
def learn_active(learner, bagging, stopping_crriterion, X_pool, X_test, y_pool, y_test):
        
    pool_size = len(y_pool)
    
    match stopping_crriterion.name:
        case "N_QUERIES":
            n_queries = N_QUERIES
        case "FRACTION_OF_POOL_QUERIES":
            n_queries = int(FRACTION_OF_POOL_QUERIES * pool_size)
        case "ENTROPY_CONFIDENCE":
            n_queries = X_pool.shape[0]
            n_decline_rounds = 0

    if n_queries * AL_QUERY_BATCH_SIZE > pool_size: # Check n_queries TODO: using as many samples as it can and only warning (?)
        raise ValueError(f"The pool samples size ({pool_size}) is to small for requested number of queries ({n_queries * AL_QUERY_BATCH_SIZE}). Decrease N_QUERIES or AL_QUERY_BATCH_SIZE.")

    # Those are sanity checks, so they are supposed to be used for code assumptions, not for user inputs checks
    # assert solver == "liblinear", ("message")

    # Evaluate initial model
    y_pred = learner.predict(X_test)
    y_proba = learner.predict_proba(X_test)

    if VERBOSE:
        print("Initial model")
        print(50*'-')

    unqueried_score = method_eval(y_test=y_test, y_pred=y_pred, y_proba=y_proba, verbose=VERBOSE)

    if VERBOSE:
        print(50*'-') 

    # AL statistics
    results = {}
    queried_classes_counter = Counter()
    performance_history = [unqueried_score]
    initial_confidence = np.mean(classifier_entropy(learner, X_test))
    entropy_confidence_history = [initial_confidence]
    filenames = []

    if VERBOSE:
        start_time = time.perf_counter()

    # Allow our model to query our unlabeled dataset for the most
    # informative points according to our query strategy (uncertainty sampling).
    for index in range(n_queries):
        query_index, _ = learner.query(X_pool, n_instances=AL_QUERY_BATCH_SIZE)

        if VERBOSE:
            print(f"{index+1}. iteration: Chosen sample was of a class: {y_pool[query_index]}")
        queried_classes_counter.update(y_pool[query_index].tolist())

        # Teach our ActiveLearner model the record it has requested.
        X, y = X_pool[query_index], y_pool[query_index]
        if not bagging:
            learner.teach(X=X, y=y)
        else:
            learner.teach(X=X, y=y, bootstrap=True, stratify=True) # for Query-By-Bagging

        # Remove the queried instance from the unlabeled pool.
        X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

        # Calculate and report our model's accuracy.
        y_pred = learner.predict(X_test)
        y_proba = learner.predict_proba(X_test)
        if VERBOSE: print(50*'-')
        model_performance = \
            method_eval(y_test=y_test, y_pred=y_pred, y_proba=y_proba, verbose=VERBOSE)
        if VERBOSE: print(50*'-')

        # Save our model's performance for plotting.
        performance_history.append(model_performance)

        # Entropy confidence
        entropy_confidence_history.append(np.mean(classifier_entropy(learner, X_test)))
        if stopping_crriterion.name == "ENTROPY_CONFIDENCE":
            if entropy_confidence_history[-1] < entropy_confidence_history[-2]:
                n_decline_rounds += 1
            else:
                n_decline_rounds = 0
            if n_decline_rounds >= N_DECLINE_ROUNDS:
                break
            if len(y_pool) < AL_QUERY_BATCH_SIZE:
                break
        
        filename = save_model(learner)
        filenames.append(filename)

    if VERBOSE:
        stop_time = time.perf_counter()
        print(f"AL querying elapsed time: {(stop_time-start_time):.2f}s")

    results["pool_training"] = {"queried_samples": queried_classes_counter, "performance_history": performance_history, "entropy_confidence": entropy_confidence_history, "filenames": filenames}
    results["final_al_classification"] = model_performance
    results["final_al_model_filepath"] = filename
    results["final_al_y_test"] = y_test.tolist()
    results["final_al_y_proba"] = y_proba[:,1].tolist()

    return results

# Main loop

class DebugLevel(Enum):
    BENCHMARK = 0
    EACH_DATASET = 1
    ACTIVE_LEARNING = 2

def test_al_methods(datasets, debug_level = 0):
    results = []

    # Active learning methods
    for al_method_name, al_method_config in active_learning_methods.items():

        # Active learning method parameters
        for i, al_method_params in enumerate(al_method_config["params"]):

            # Classificator models
            for j, (classificator_name, classificator_config) in enumerate(al_method_config["classifiers"].items()):

                results.append({"al_method_name": str(al_method_name),
                                "al_method_params": al_method_params,
                                "classificator_name": str(classificator_name),
                                "classificator_config": classificator_config,
                                "datasets_results": []
                                })

                all_datasets_accuracy_arr = []
                all_datasets_precision_arr = []
                all_datasets_recall_arr = []
                all_datasets_f2_score_arr = []
                all_datasets_auc_pr_curve_arr = [] 
                all_datasets_auc_roc_curve_arr = []

                # Datasets
                for dataset_name, dataset in datasets.items():

                    if VERBOSE:
                        print(50*"-"+"\n",\
                            f"Active learning method: {al_method_name}\n",\
                            f"Active leraning parameters: {al_method_params}\n",\
                            f"Classificator: {classificator_name}\n",\
                            f"Classificator config: {classificator_config}\n",\
                            f"Dataset: {dataset_name}\n",\
                            50*"-"+"\n", sep='', end='\n')

                    # Standardizing
                    dataset = standardize(dataset)

                    # Spliting
                    X, y = split_dataset_xy(dataset)

                    dataset_info = {
                        "samples_no": len(y), "class_counter": Counter(y.tolist())
                    }

                    al_results_arr = []
                    folds_final_models_results = []
                    folds_final_models_tests = []
                    folds_final_models_probas = []

                    if VERBOSE:
                        start_time = time.perf_counter()

                    # Stratified K-Folds cross-validator
                    skf = StratifiedKFold(n_splits=int(1/TEST_SIZE), random_state=RANDOM_STATE_SEED, shuffle=True) 
                    for k, (kf_train_indices, kf_test_indices) in enumerate(skf.split(X, y)):

                        fold_results = {}

                        # K-Fold train and test
                        X_train, X_test, y_train, y_test = X.iloc[kf_train_indices], \
                        X.iloc[kf_test_indices], y.iloc[kf_train_indices], y.iloc[kf_test_indices]
                        
                        # K-Fold pool from train
                        X_train_init, X_pool, y_train_init, y_pool = train_test_split(X_train, y_train, \
                                                                        train_size=INITIAL_TRAIN_SIZE, stratify=y_train, random_state=RANDOM_STATE_SEED)
                        
                        # Coversion to numpy
                        X_train = X_train.to_numpy()
                        y_train = y_train.values
                        X_train_init = X_train_init.to_numpy()
                        y_train_init = y_train_init.values
                        X_pool = X_pool.to_numpy()
                        y_pool = y_pool.values
                        X_test = X_test.to_numpy()
                        y_test = y_test.values

                        fold_results["splits"] = {
                            "train": {"samples_no": len(y_train), "class_counter": Counter(y_train.tolist())}, 
                            "test": {"samples_no": len(y_test), "class_counter": Counter(y_test.tolist())}, 
                            "train_init": {"samples_no": len(y_train_init), "class_counter": Counter(y_train_init.tolist())}, 
                            "pool": {"samples_no": len(y_pool), "class_counter": Counter(y_pool.tolist())}
                        }

                        if VERBOSE:
                            print(f"Dataset fold no. {k+1} splitting: ")
                            _ = [print(x) for x in fold_results["splits"].items()]

                        # Preparing the ActiveLearner
                        learner = None
                        params = None
                        # Query-by-Bagging
                        bagging = False
                        if al_method_name.startswith("query_by"): # create Committee
                            learners = []
                            full_learners = []
                            bagging = True if al_method_name == "query_by_bagging" else False
                            # Query-By-Committee homogenous
                            if classificator_config["committee_type"] == "homogenous":
                                Clf = classificator_config["Clf"]
                                params = tune_classifiers_hyperparameters(Clf, X_train_init, y_train_init, PARAMS_GRID[Clf.__name__], GridSearchCV, f2_score, verbose=False)
                                for _ in range(classificator_config["n_models"]):
                                    learner = ActiveLearner(
                                            estimator = Clf(**params),
                                            X_training = X_train_init, 
                                            y_training = y_train_init,
                                            bootstrap_init = bagging,
                                            stratify = bagging
                                        )
                                    learners.append(learner)
                                    if FULL_LEARNER_COMPARISON:
                                        full_learner = ActiveLearner(
                                                estimator = Clf(**params),
                                                X_training = X_train, 
                                                y_training = y_train,
                                                bootstrap_init = bagging,
                                                stratify = bagging
                                            )
                                        full_learners.append(full_learner)
                            # Query-By-Committee heterogenous
                            elif classificator_config["committee_type"] == "heterogenous":
                                params = []
                                for CommitteeClf in classificator_config["Clfs"]:
                                    clf_params = tune_classifiers_hyperparameters(CommitteeClf, X_train_init, y_train_init, PARAMS_GRID[CommitteeClf.__name__], GridSearchCV, f2_score, verbose=False)
                                    learner = ActiveLearner(
                                            estimator = CommitteeClf(**clf_params),
                                            X_training = X_train_init, 
                                            y_training = y_train_init,
                                            bootstrap_init = bagging,
                                            stratify = bagging
                                        )
                                    learners.append(learner)
                                    params.append(clf_params)
                                    if FULL_LEARNER_COMPARISON:
                                        full_learner = ActiveLearner(
                                            estimator = CommitteeClf(**clf_params),
                                            X_training = X_train, 
                                            y_training = y_train,
                                            bootstrap_init = bagging,
                                            stratify = bagging
                                        )
                                        full_learners.append(full_learner)
                            learner = Committee(
                                    learner_list = learners,
                                    query_strategy = al_method_params["query_strategy"]
                                )
                            if FULL_LEARNER_COMPARISON:
                                full_learner = Committee(
                                    learner_list = full_learners,
                                    query_strategy = al_method_params["query_strategy"]
                                )
                        # Single model ActiveLearner
                        else:
                            Clf = classificator_config["Clf"]
                            if classificator_config["params"] == "gs":
                                params = tune_classifiers_hyperparameters(Clf, X_train_init, y_train_init, PARAMS_GRID[Clf.__name__], GridSearchCV, f2_score, verbose=False)
                            else:
                                params = classificator_config["params"]
                            learner = ActiveLearner(
                                estimator=Clf(**params), X_training = X_train_init, 
                                y_training = y_train_init, query_strategy=al_method_params["query_strategy"]
                            )
                            if FULL_LEARNER_COMPARISON:
                                full_learner = ActiveLearner(
                                    estimator=Clf(**params), X_training = X_train, 
                                    y_training = y_train, query_strategy=al_method_params["query_strategy"]
                                )
                        
                        fold_results["classifier_params"] = params
                        
                        # Model trained on whole pool comparison
                        if FULL_LEARNER_COMPARISON:
                            y_pred = full_learner.predict(X_test)
                            y_proba = full_learner.predict_proba(X_test)

                            if VERBOSE:
                                print("Full train dataset model")
                                print(50*'-')

                            full_model_score = \
                                method_eval(y_test=y_test, y_pred=y_pred, y_proba=y_proba, verbose=VERBOSE)
                            
                            if VERBOSE:
                                print(50*'-')

                            fold_results["full_train_classification"] = full_model_score
                            filename = save_model(full_learner)
                            fold_results["full_train_filepath"] = filename

                        # Learn actively!
                        al_results = \
                            learn_active(learner=learner, bagging=bagging, stopping_crriterion=al_method_params["stopping_criterion"], \
                                           X_pool=X_pool, X_test=X_test, \
                                           y_pool=y_pool, y_test=y_test)
                        
                        fold_results["al_classification"] = al_results

                        folds_final_models_results.append(al_results["final_al_classification"])
                        folds_final_models_tests.append(al_results["final_al_y_test"])
                        folds_final_models_probas.append(al_results["final_al_y_proba"])

                        al_results_arr.append(fold_results)


                    if VERBOSE:
                        print(50*"-"+"\n",\
                            f"Active learning method: {al_method_name}\n",\
                            f"Active leraning parameters: {al_method_params}\n",\
                            f"Classificator: {classificator_name}\n",\
                            f"Classificator config: {classificator_config}\n",\
                            f"Dataset: {dataset_name}\n",\
                            50*"-"+"\n", sep='', end='\n')
                        
                    stop_time = time.perf_counter()

                    # Result for benchmark dataset - mean of kfolds
                    mean_accuracy = np.mean([x["metrics"]["accuracy"] for x in folds_final_models_results])
                    mean_precision = np.mean([x["metrics"]["precision"] for x in folds_final_models_results])
                    mean_recall = np.mean([x["metrics"]["recall"] for x in folds_final_models_results])
                    mean_f2_score = np.mean([x["metrics"]["f2_score"] for x in folds_final_models_results])
                    mean_auc_pr_curve = np.mean([x["metrics"]["auc_pr_curve"] for x in folds_final_models_results])
                    mean_auc_roc_curve = np.mean([x["metrics"]["auc_roc_curve"] for x in folds_final_models_results])
                    # mean_g_mean = np.mean([x["metrics"]["g_mean"] for x in folds_final_models_results])

                    folds_final_models_tests_concat = np.concatenate(folds_final_models_tests)
                    folds_final_models_probas_concat = np.concatenate(folds_final_models_probas)

                    _, pr_plot_filename = plot_pr_curve(folds_final_models_tests_concat, folds_final_models_probas_concat)
                    _, roc_plot_filename = plot_roc(folds_final_models_tests_concat, folds_final_models_probas_concat)

                    if VERBOSE:
                        print(
                            f"Accuracy: {mean_accuracy:.2f}\n",\
                            f"Precision: {mean_precision:.2f}\n",\
                            f"Recall: {mean_recall:.2f}\n",\
                            f"F2 score: {mean_f2_score:.2f}\n",\
                            f"Precision-Recall AUC: {mean_auc_pr_curve:.2f}\n",\
                            f"ROC AUC: {mean_auc_roc_curve:.2f}\n",\
                            # f"G mean: {mean_g_mean:.2f}\n", sep='', \
                            end='\n')
                        print(f"Dataset all folds time: {(stop_time-start_time):.2f}s")

                    dataset_result = {"dataset_name": dataset_name,
                                        "dataset_info": dataset_info,
                                        "folds":  al_results_arr,
                                        "kfoldcv": {
                                                "accuracy": mean_accuracy,
                                                "precision": mean_precision,
                                                "recall": mean_recall,
                                                "f2_score": mean_f2_score,
                                                "auc_pr_curve": mean_auc_pr_curve,
                                                "auc_roc_curve": mean_auc_roc_curve,
                                                "folds_final_models_tests_concat": folds_final_models_tests_concat.tolist(),
                                                "folds_final_models_probas_concat" : folds_final_models_probas_concat.tolist(),
                                                "pr_curve_fig_filename": pr_plot_filename,
                                                "roc_curve_fig_filename": roc_plot_filename,
                                                # "g_mean": mean_g_mean
                                                }
                    }

                    filepath = PARTIAL_RESULTS_PATH + dataset_name + '_' + al_method_name + f'_al_params_{i}_' + f"classifier_{j}.json"
                    with open(filepath, "w") as outfile:
                        json.dump(dataset_result, outfile)
                    
                    results[-1]["datasets_results"].append(dataset_result)
                    
                    all_datasets_accuracy_arr.append(mean_accuracy)
                    all_datasets_precision_arr.append(mean_precision)
                    all_datasets_recall_arr.append(mean_recall)
                    all_datasets_f2_score_arr.append(mean_f2_score)
                    all_datasets_auc_pr_curve_arr.append(mean_auc_pr_curve)
                    all_datasets_auc_roc_curve_arr.append(mean_auc_roc_curve)
                    # all_datasets_g_mean.append(mean_g_mean)
                
                # Result for all benchmark datasets - mean of all benchmark datasets
                mean_accuracy = np.mean(all_datasets_accuracy_arr)
                mean_precision = np.mean(all_datasets_precision_arr)
                mean_recall = np.mean(all_datasets_recall_arr)
                mean_f2_score = np.mean(all_datasets_f2_score_arr)
                mean_auc_pr_curve = np.mean(all_datasets_auc_pr_curve_arr)
                mean_auc_roc_curve = np.mean(all_datasets_auc_roc_curve_arr)
                # mean_g_mean = np.mean(all_datasets_g_mean)

                results[-1]["benchmark_result"] = {
                    "accuracy": mean_accuracy,
                    "precision": mean_precision,
                    "recall": mean_recall,
                    "f2_score": mean_f2_score,
                    "auc_pr_curve": mean_auc_pr_curve,
                    "auc_roc_curve": mean_auc_roc_curve,
                    # "g_mean": mean_g_mean
                }
                
                if VERBOSE:
                    print(50*"-"+"\n",\
                            f"Active learning method: {al_method_name}\n",\
                            f"Active learning parameters: {al_method_params}\n",\
                            f"Classificator name: {classificator_name}\n",\
                            f"Classificator config: {classificator_config}\n",\
                            f"All datasets\n",\
                            50*"-"+"\n", sep='', end='\n')
                
                    print(f"Accuracy: {mean_accuracy:.2f}\n",\
                        f"Precision: {mean_precision:.2f}\n",\
                        f"Recall: {mean_recall:.2f}\n",\
                        f"F2 score: {mean_f2_score:.2f}\n",\
                        f"Precision-Recall AUC: {mean_auc_pr_curve:.2f}\n",\
                        f"ROC AUC: {mean_auc_roc_curve:.2f}\n",\
                        # f"G mean: {mean_g_mean:.2f}\n", sep='', 
                        end='\n')
                
    return results