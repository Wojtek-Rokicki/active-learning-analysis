# Classification score
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc, precision_score, recall_score, fbeta_score, make_scorer
from imblearn.metrics import geometric_mean_score

def method_eval(y_test, y_pred, y_proba, verbose=0):

    results = {}

    accuracy = accuracy_score(y_test, y_pred)

    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba[:, 1]) # positive when >= threshold
    auc_roc_curve = auc(fpr, tpr)

    # Plotting ROC
    # pyplot.plot(fpr, tpr, marker='.', label='ROC')

    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_proba[:, 1]) # positive when >= threshold
    auc_pr_curve = auc(recalls, precisions)

    # Plotting PR Curve
    # pyplot.plot(recalls, precisions, marker='.', label='ROC')

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    # Fbeta = ((1 + beta^2) * Precision * Recall) / (beta^2 * Precision + Recall)

    if precision+recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        f2_score = 5 * precision * recall / (4 * precision + recall)
    else:
        f1_score = 0
        f2_score = 0

    g_mean = geometric_mean_score(y_test, y_pred)

    results["metrics"] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                # "f1_score": f1_score,
                "f2_score": f2_score,
                "auc_pr_curve": auc_pr_curve,
                "auc_roc_curve": auc_roc_curve,
                # "g_mean": g_mean
            }
    
    results["curves"] = {
        "pr_curve":{"precisions": precisions.tolist(), "recalls": recalls.tolist(), "pr_thresholds": pr_thresholds.tolist()},
        "roc_curve":{"fpr": fpr.tolist(), "tpr": tpr.tolist(), "roc_thresholds": roc_thresholds.tolist()}
        }
    
    if verbose:
        print(
            f"accuracy: {accuracy},",
            f"precision: {precision},",
            f"recall: {recall},",
            # f"f1_score: {f1_score},",
            f"f2_score: {f2_score},",
            f"auc_pr_curve: {auc_pr_curve},",
            f"auc_roc_curve: {auc_roc_curve}",
            # f"g_mean: {g_mean},", 
            sep="\n")

    return results

f2_func = lambda y_true, y_pred: fbeta_score(y_true=y_true, y_pred=y_pred, beta=2, zero_division=0)

def pr_auc_func(y_true, y_pred):
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred)
    auc_pr_curve = auc(recalls, precisions)
    return auc_pr_curve

# for ROC AUC just type ‘roc_auc’
pr_auc_score = make_scorer(pr_auc_func, greater_is_better=True)
f2_score = make_scorer(f2_func, greater_is_better=True)