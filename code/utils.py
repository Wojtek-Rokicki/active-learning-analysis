import json
import numpy as np
from config import DATASETS_NAMES, ACTIVE_LEARNING_METHODS, CLASSIFIERS, PARTIAL_RESULTS_PATH

def aggregate_n_kcv_metrics(results: dict) -> dict:

    # Result template
    result = {
                "al_classification":
                {
                    "mean":
                    {
                        "accuracy": None,
                        "precision": None,
                        "recall": None,
                        "f2_score": None,
                        "auc_pr_curve": None,
                        "auc_roc_curve": None
                    },
                    "var":
                    {
                        "accuracy": None,
                        "precision": None,
                        "recall": None,
                        "f2_score": None,
                        "auc_pr_curve": None,
                        "auc_roc_curve": None,
                    },
                    "entropy_confidence": []
                },
                "full_train_classification":
                {
                    "mean":
                    {
                        "accuracy": None,
                        "precision": None,
                        "recall": None,
                        "f2_score": None,
                        "auc_pr_curve": None,
                        "auc_roc_curve": None
                    },
                    "var":
                    {
                        "accuracy": None,
                        "precision": None,
                        "recall": None,
                        "f2_score": None,
                        "auc_pr_curve": None,
                        "auc_roc_curve": None
                    },
                }
            }

    # Aggregate across folds
    nkcv_al_mean_acc = []
    nkcv_al_mean_prec = []
    nkcv_al_mean_rec = []
    nkcv_al_mean_f2 = []
    nkcv_al_mean_pr_auc = []
    nkcv_al_mean_roc_auc = []

    nkcv_al_var_acc = []
    nkcv_al_var_prec = []
    nkcv_al_var_rec = []
    nkcv_al_var_f2 = []
    nkcv_al_var_pr_auc = []
    nkcv_al_var_roc_auc = []

    nkcv_al_mean_entropy_confidence = []

    nkcv_full_mean_acc = []
    nkcv_full_mean_prec = []
    nkcv_full_mean_rec = []
    nkcv_full_mean_f2 = []
    nkcv_full_mean_pr_auc = []
    nkcv_full_mean_roc_auc = []

    nkcv_full_var_acc = []
    nkcv_full_var_prec = []
    nkcv_full_var_rec = []
    nkcv_full_var_f2 = []
    nkcv_full_var_pr_auc = []
    nkcv_full_var_roc_auc = []

    # For each n'th kcv
    for kcv in results["n_kcv_results"]:

        # Aggregate across folds
        folds_al_agg_acc = []
        folds_al_agg_prec = []
        folds_al_agg_rec = []
        folds_al_agg_f2 = []
        folds_al_agg_pr_auc = []
        folds_al_agg_roc_auc = []

        folds_al_agg_entropy_confidence = []

        folds_full_agg_acc = []
        folds_full_agg_prec = []
        folds_full_agg_rec = []
        folds_full_agg_f2 = []
        folds_full_agg_pr_auc = []
        folds_full_agg_roc_auc = []

        # For each k'th fold
        for kthfold in kcv["kcv_results"]:

            # Aggregate individual metrics
            inner_agg_acc = []
            inner_agg_prec = []
            inner_agg_rec = []
            inner_agg_f2 = []
            inner_agg_pr_auc = []
            inner_agg_roc_auc = []

            # For each iteration
            for al_iter in kthfold["al_classification"]["performance_history"]:
                inner_agg_acc.append(al_iter["metrics"]["accuracy"])
                inner_agg_prec.append(al_iter["metrics"]["precision"])
                inner_agg_rec.append(al_iter["metrics"]["recall"])
                inner_agg_f2.append(al_iter["metrics"]["f2_score"])
                inner_agg_pr_auc.append(al_iter["metrics"]["auc_pr_curve"])
                inner_agg_roc_auc.append(al_iter["metrics"]["auc_roc_curve"])

            folds_al_agg_acc.append(inner_agg_acc)
            folds_al_agg_prec.append(inner_agg_prec)
            folds_al_agg_rec.append(inner_agg_rec)
            folds_al_agg_f2.append(inner_agg_f2)
            folds_al_agg_pr_auc.append(inner_agg_pr_auc)
            folds_al_agg_roc_auc.append(inner_agg_roc_auc)

            folds_al_agg_entropy_confidence.append(kthfold["al_classification"]["entropy_confidence"])

            folds_full_agg_acc.append(kthfold["full_train_classification"]["metrics"]["accuracy"])
            folds_full_agg_prec.append(kthfold["full_train_classification"]["metrics"]["precision"])
            folds_full_agg_rec.append(kthfold["full_train_classification"]["metrics"]["recall"])
            folds_full_agg_f2.append(kthfold["full_train_classification"]["metrics"]["f2_score"])
            folds_full_agg_pr_auc.append(kthfold["full_train_classification"]["metrics"]["auc_pr_curve"])
            folds_full_agg_roc_auc.append(kthfold["full_train_classification"]["metrics"]["auc_roc_curve"])

        nkcv_al_mean_acc.append(list(map(lambda x: np.mean(x), zip(*folds_al_agg_acc))))
        nkcv_al_mean_prec.append(list(map(lambda x: np.mean(x), zip(*folds_al_agg_prec))))
        nkcv_al_mean_rec.append(list(map(lambda x: np.mean(x), zip(*folds_al_agg_rec))))
        nkcv_al_mean_f2.append(list(map(lambda x: np.mean(x), zip(*folds_al_agg_f2))))
        nkcv_al_mean_pr_auc.append(list(map(lambda x: np.mean(x), zip(*folds_al_agg_pr_auc))))
        nkcv_al_mean_roc_auc.append(list(map(lambda x: np.mean(x), zip(*folds_al_agg_roc_auc))))

        nkcv_al_var_acc.append(list(map(lambda x: np.var(x), zip(*folds_al_agg_acc))))
        nkcv_al_var_prec.append(list(map(lambda x: np.var(x), zip(*folds_al_agg_prec))))
        nkcv_al_var_rec.append(list(map(lambda x: np.var(x), zip(*folds_al_agg_rec))))
        nkcv_al_var_f2.append(list(map(lambda x: np.var(x), zip(*folds_al_agg_f2))))
        nkcv_al_var_pr_auc.append(list(map(lambda x: np.var(x), zip(*folds_al_agg_pr_auc))))
        nkcv_al_var_roc_auc.append(list(map(lambda x: np.var(x), zip(*folds_al_agg_roc_auc))))

        nkcv_al_mean_entropy_confidence.append(list(map(lambda x: np.mean(x), zip(*folds_al_agg_entropy_confidence))))

        nkcv_full_mean_acc.append(np.mean(folds_full_agg_acc))
        nkcv_full_mean_prec.append(np.mean(folds_full_agg_prec))
        nkcv_full_mean_rec.append(np.mean(folds_full_agg_rec))
        nkcv_full_mean_f2.append(np.mean(folds_full_agg_f2))
        nkcv_full_mean_pr_auc.append(np.mean(folds_full_agg_pr_auc))
        nkcv_full_mean_roc_auc.append(np.mean(folds_full_agg_roc_auc))

        nkcv_full_var_acc.append(np.var(folds_full_agg_acc))
        nkcv_full_var_prec.append(np.var(folds_full_agg_prec))
        nkcv_full_var_rec.append(np.var(folds_full_agg_rec))
        nkcv_full_var_f2.append(np.var(folds_full_agg_f2))
        nkcv_full_var_pr_auc.append(np.var(folds_full_agg_pr_auc))
        nkcv_full_var_roc_auc.append(np.var(folds_full_agg_roc_auc))

    result["al_classification"]["mean"]["accuracy"] = list(map(lambda x: np.mean(x), zip(*nkcv_al_mean_acc)))
    result["al_classification"]["mean"]["precision"] = list(map(lambda x: np.mean(x), zip(*nkcv_al_mean_prec)))
    result["al_classification"]["mean"]["recall"] = list(map(lambda x: np.mean(x), zip(*nkcv_al_mean_rec)))
    result["al_classification"]["mean"]["f2_score"] = list(map(lambda x: np.mean(x), zip(*nkcv_al_mean_f2)))
    result["al_classification"]["mean"]["auc_pr_curve"] = list(map(lambda x: np.mean(x), zip(*nkcv_al_mean_pr_auc)))
    result["al_classification"]["mean"]["auc_roc_curve"] = list(map(lambda x: np.mean(x), zip(*nkcv_al_mean_roc_auc)))

    result["al_classification"]["var"]["accuracy"] = list(map(lambda x: np.mean(x), zip(*nkcv_al_var_acc)))
    result["al_classification"]["var"]["precision"] = list(map(lambda x: np.mean(x), zip(*nkcv_al_var_prec)))
    result["al_classification"]["var"]["recall"] = list(map(lambda x: np.mean(x), zip(*nkcv_al_var_rec)))
    result["al_classification"]["var"]["f2_score"] = list(map(lambda x: np.mean(x), zip(*nkcv_al_var_f2)))
    result["al_classification"]["var"]["auc_pr_curve"] = list(map(lambda x: np.mean(x), zip(*nkcv_al_var_pr_auc)))
    result["al_classification"]["var"]["auc_roc_curve"] = list(map(lambda x: np.mean(x), zip(*nkcv_al_var_roc_auc)))

    result["al_classification"]["entropy_confidence"] = list(map(lambda x: np.mean(x), zip(*nkcv_al_mean_entropy_confidence)))

    result["full_train_classification"]["mean"]["accuracy"] = np.mean(nkcv_full_mean_acc)
    result["full_train_classification"]["mean"]["precision"] = np.mean(nkcv_full_mean_prec)
    result["full_train_classification"]["mean"]["recall"] = np.mean(nkcv_full_mean_rec)
    result["full_train_classification"]["mean"]["f2_score"] = np.mean(nkcv_full_mean_f2)
    result["full_train_classification"]["mean"]["auc_pr_curve"] = np.mean(nkcv_full_mean_pr_auc)
    result["full_train_classification"]["mean"]["auc_roc_curve"] = np.mean(nkcv_full_mean_roc_auc)

    result["full_train_classification"]["var"]["accuracy"] = np.mean(nkcv_full_var_acc)
    result["full_train_classification"]["var"]["precision"] = np.mean(nkcv_full_var_prec)
    result["full_train_classification"]["var"]["recall"] = np.mean(nkcv_full_var_rec)
    result["full_train_classification"]["var"]["f2_score"] = np.mean(nkcv_full_var_f2)
    result["full_train_classification"]["var"]["auc_pr_curve"] = np.mean(nkcv_full_var_pr_auc)
    result["full_train_classification"]["var"]["auc_roc_curve"] = np.mean(nkcv_full_var_roc_auc)

    return result


def read_json_dump(file_path: str) -> dict:
    json_dump = json.load(open(file_path))
    return json_dump

def read_all_n_kcv_agg_results():

    datasets_names = DATASETS_NAMES
    al_methods_names = ACTIVE_LEARNING_METHODS.keys()
    classifiers_names = [clf_config[0].__name__ for clf_config in CLASSIFIERS]

    results = dict()
    for dataset_name in datasets_names:
        for classifiers_name in classifiers_names:
            results[dataset_name + '-' + classifiers_name] = []

    for dataset_name in datasets_names:
        for classifiers_name in classifiers_names:
            for al_method_name in al_methods_names:
                result_file_path = PARTIAL_RESULTS_PATH / dataset_name / classifiers_name / al_method_name
                result = read_json_dump(str(result_file_path) + "_n_kcv_agg")
                results[dataset_name + '-' + classifiers_name].append({al_method_name: result})
    
    return results

def aggregate_saved_results():
    datasets_names = DATASETS_NAMES
    al_methods_names = ACTIVE_LEARNING_METHODS.keys()
    classifiers_names = [clf_config[0].__name__ for clf_config in CLASSIFIERS]

    for dataset_name in datasets_names:
        for classifiers_name in classifiers_names:
            for al_method_name in al_methods_names:
                result_file_path = PARTIAL_RESULTS_PATH / dataset_name / classifiers_name / al_method_name
                result = read_json_dump(str(result_file_path)  + "_n_kcv")
                agg_result = aggregate_n_kcv_metrics(result)
                json.dump(agg_result, open(str(result_file_path) + f"_n_kcv_agg",'w'))
