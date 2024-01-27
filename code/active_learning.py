import time, json, math

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

from config import *
from preprocessing import standardize, split_dataset_xy
from metrics import method_eval
from utils import aggregate_n_kcv_metrics

from modAL import ActiveLearner
from modAL.uncertainty import classifier_entropy # TODO: hide classifier entropy in method_eval

# Active Learning querying
def learn_active(learner, X_pool, X_test, y_pool, y_test):
        
    pool_size = len(y_pool)

    # Prepare number of queries and batch size
    if learner.query_strategy.__name__ in COMPUTATIONALLY_COMPLEX_METHODS and \
        pool_size > 6000:
        batch_size = math.floor((1/1000) * pool_size)
        n_queries = math.ceil(pool_size / batch_size)
    else:
        n_queries = pool_size
        batch_size = 1

    # Evaluate initial model
    y_pred = learner.predict(X_test)
    y_proba = learner.predict_proba(X_test)

    unqueried_score = method_eval(y_test=y_test, y_pred=y_pred, y_proba=y_proba, verbose=False, curves=False)

    # AL statistics
    results = {}
    performance_history = [unqueried_score]
    initial_confidence = np.mean(classifier_entropy(learner, X_test))
    entropy_confidence_history = [initial_confidence]

    # Start AL querying
    for index in range(n_queries):
        query_index, _ = learner.query(X_pool, n_instances=batch_size)

        # Teach our ActiveLearner model the record it has requested.
        X, y = X_pool[query_index], y_pool[query_index]
        learner.teach(X=X, y=y)

        # Remove the queried instance from the unlabeled pool.
        X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

        # Calculate our model's metrics.
        y_pred = learner.predict(X_test)
        y_proba = learner.predict_proba(X_test)
        model_performance = \
            method_eval(y_test=y_test, y_pred=y_pred, y_proba=y_proba, verbose=False, curves=False)

        performance_history.append(model_performance)
        entropy_confidence_history.append(np.mean(classifier_entropy(learner, X_test)))

    results = {"performance_history": performance_history, "entropy_confidence": entropy_confidence_history}

    return results

# Main loop
def test_al_methods(datasets: dict):

    # Dataset
    for dataset_name, dataset in datasets.items():

        # Active learning method 
        for al_method_name, al_method_config in ACTIVE_LEARNING_METHODS.items():

            # Classificator model
            for classificator, classificator_params in al_method_config["classifiers"]:              

                if VERBOSE:
                    print(50*"-"+"\n",\
                        f"Dataset: {dataset_name}\n",\
                        f"Classificator: {classificator.__name__}\n",\
                        f"Active learning method: {al_method_name}\n",\
                        50*"-"+"\n", sep='', end='\n')

                # Standardizing
                dataset = standardize(dataset)

                # Spliting
                X, y = split_dataset_xy(dataset)

                n_kcv_results = {'n_kcv_results': []}

                if VERBOSE:
                    nkcv_start_time = time.perf_counter()

                # N x k-CV
                for n in range(N_KCV):

                    kfold_results = {"kcv_results": []}

                    if VERBOSE:
                        kcv_start_time = time.perf_counter()

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
                        
                        # Coversion to numpy (mainly for modAL)
                        X_train = X_train.to_numpy()
                        y_train = y_train.values
                        X_train_init = X_train_init.to_numpy()
                        y_train_init = y_train_init.values
                        X_pool = X_pool.to_numpy()
                        y_pool = y_pool.values
                        X_test = X_test.to_numpy()
                        y_test = y_test.values

                        if VERBOSE:
                            print(f"Fold no. {k+1} ...")

                        # Preparing the ActiveLearner
                        learner = ActiveLearner(
                                            estimator = classificator(**classificator_params),
                                            X_training = X_train_init, 
                                            y_training = y_train_init
                                        )

                        # Model trained on whole pool comparison
                        if FULL_LEARNER_COMPARISON:
                            full_learner = classificator(**classificator_params)
                            full_learner.fit(X_train, y_train)
                            y_pred = full_learner.predict(X_test)
                            y_proba = full_learner.predict_proba(X_test)

                            full_model_score = \
                                method_eval(y_test=y_test, y_pred=y_pred, y_proba=y_proba, verbose=False)

                            fold_results["full_train_classification"] = full_model_score

                        if VERBOSE:
                            k_start_time = time.perf_counter()

                        # Learn actively!
                        al_results = \
                            learn_active(learner=learner, \
                                            X_pool=X_pool, X_test=X_test, \
                                            y_pool=y_pool, y_test=y_test)
                        
                        # Time of k'th fold of k-CV
                        if VERBOSE:
                            k_stop_time = time.perf_counter()
                            print(f"Fold no. {k+1} took {(k_stop_time-k_start_time):.4f}s")

                        fold_results["al_classification"] = al_results
                        kfold_results['kcv_results'].append(fold_results)


                    # Time of n'th k-CV
                    if VERBOSE:
                        kcv_stop_time = time.perf_counter()
                        print(f'N = {n+1} took {(kcv_stop_time-kcv_start_time):.4f}s')

                    n_kcv_results['n_kcv_results'].append(kfold_results)

                # Time of N x k-CV
                if VERBOSE:
                    nkcv_stop_time = time.perf_counter()
                    print(f'N x k-CV took {(nkcv_stop_time-nkcv_start_time):.4f}s')

                # Save results
                json.dump(n_kcv_results, open(PARTIAL_RESULTS_PATH / f"{dataset_name}_{al_method_name}_{classificator.__name__}_n_kcv",'w'))
                    
                # Aggregate metrics across all folds
                aggregated_metrics = aggregate_n_kcv_metrics(n_kcv_results)

                # Extract metrics for 10, 25, 50, 100 percentage of train dataset
                # TODO
                    
                # Save aggregated results
                json.dump(aggregated_metrics, open(PARTIAL_RESULTS_PATH / f"{dataset_name}_{al_method_name}_{classificator.__name__}_n_kcv_agg",'w'))

                results = aggregated_metrics

    return results