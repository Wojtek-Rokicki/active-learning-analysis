import uuid, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from config import RANDOM_STATE_SEED, PLOTS_PATH, INITIAL_TRAIN_SIZE

try:
    os.makedirs(PLOTS_PATH)
except FileExistsError:
    pass

# PCA decomposition
from sklearn.decomposition import PCA

def plot_pca(dataset_name, df, random_state=RANDOM_STATE_SEED):
    # Set our RNG seed for reproducibility.
    np.random.seed(random_state)

    X_raw = df.loc[:, df.columns != 'target'].to_numpy()
    y_raw = df['target'].to_numpy()

    # Define our PCA transformer and fit it onto our raw dataset.
    pca = PCA(n_components=2, random_state=RANDOM_STATE_SEED)
    transformed_df = pca.fit_transform(X=X_raw)

    # Isolate the data we'll need for plotting.
    x_component, y_component = transformed_df[:, 0], transformed_df[:, 1]

    # Plot our dimensionality-reduced (via PCA) dataset.
    plt.figure(figsize=(8.5, 6), dpi=130)
    plt.scatter(x=x_component, y=y_component, c=y_raw, cmap='viridis', s=50, alpha=8/10)
    plt.title(f'{dataset_name} classes after PCA transformation')
    plt.savefig(f"{dataset_name}_pca.png") # TODO: Add the path to results
    plt.show()

# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from modAL import Committee

def plot_confusion_matrix(learner, y_test, y_pred):
    if not issubclass(learner.__class__, Committee):
        classes = learner.estimator.classes_
    else:
        classes = learner.classes_

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    disp.plot()
    plt.show(block=False)

from sklearn.metrics import roc_curve

def plot_roc(y_test, y_probas):
    fpr, tpr, _ = roc_curve(y_test, y_probas)
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    fig = plt.gcf()
    unique_filename = str(uuid.uuid4())+'.png'
    fig.savefig(PLOTS_PATH + unique_filename)
    # plt.show(block=False)
    plt.close()
    return fig, unique_filename

from sklearn.metrics import precision_recall_curve
    
def plot_pr_curve(y_test, y_probas):
    precisions, recalls, _ = precision_recall_curve(y_test, y_probas)
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    fig = plt.gcf()
    unique_filename = str(uuid.uuid4())+'.png'
    fig.savefig(PLOTS_PATH + unique_filename)
    # plt.show(block=False)
    plt.close()
    return fig, unique_filename

AL_LABEL_NAME_MAPPING = {
    "random_sampling": "RS",
    "uncertainty_sampling": "US",
    "expected_error_reduction_01": "EER_01",
    "expected_error_reduction_log": "EER_LOG",
    "variance_reduction": "VR"
}

METRIC_TITLE_NAME_MAPPING = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f2_score": "F2 Score",
    "auc_pr_curve": "PR AUC",
    "auc_roc_curve": "ROC AUC",
    "entropy_confidence": None
}

# Plot metric on axis
def plot_aggregated_metric(ax: Axis, X, y, label=None) -> Axis:
    ax.plot(X, y, label=label)
    ax.set(xlim=(0, 100))
    return ax

# Plot all 6 metrics of al method
def plot_al_method_all_metrics(axs: Axes, al_method_name: str, al_method_metrics: dict) -> Axes:

    n_queries = len(list(al_method_metrics.values())[0])

    for i, metric_name in enumerate(al_method_metrics.keys()):
        
        ax = axs[int(i/3), int(i%3)]
        x_axis = [(x/n_queries)*((1-INITIAL_TRAIN_SIZE)*100) + INITIAL_TRAIN_SIZE*100 for x in range(n_queries)]
        ax = plot_aggregated_metric(ax, x_axis, al_method_metrics[metric_name], label=AL_LABEL_NAME_MAPPING[al_method_name])
        
        # Title and label setup
        ax.set_title(METRIC_TITLE_NAME_MAPPING[metric_name])
        ax.legend(loc='lower right')
        
        # Ticks setup
        # ax.xaxis.set_major_formatter(mtick.PercentFormatter()) 
        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

    return axs

def plot_full_model_metric(ax: Axis, y, x_len: int) -> Axes:
    handle = ax.plot(range(x_len), np.ones(x_len)*y, linewidth=2, color='r')
    return handle

# Plot all 6 metrics of full model
def plot_full_model_metrics(axs: Axes, full_model_metrics: dict, x_len: int) -> Axes:
    for i, metric_name in enumerate(full_model_metrics.keys()):
        ax = axs[int(i/3), int(i%3)]
        handle = plot_full_model_metric(ax, full_model_metrics[metric_name], x_len)
    return axs, handle


def plot_all_n_kcv_agg_all_metrics_results(results: dict):
    for dataset_classifier_combination_label, n_kcv_als_results in results.items():
        dataset_name, classifier_name = dataset_classifier_combination_label.split('-')

        fig, axs = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True)
        for n_kcv_al_result in n_kcv_als_results:
            al_method_name = list(n_kcv_al_result.keys())[0]
            al_method_result = list(n_kcv_al_result.values())[0]
            axs = plot_al_method_all_metrics(axs, al_method_name, al_method_result["al_classification"]["mean"])
        
        axs, handle = plot_full_model_metrics(axs, al_method_result["full_train_classification"]["mean"], len(al_method_result["al_classification"]["mean"]["accuracy"]))
        
        fig.suptitle(f"Zapytania aktywnego uczenia \n dla klasyfikatora {classifier_name} na zbiorze {dataset_name}", fontsize=16)
        fig.supxlabel("% zbioru trenującego")
        fig.supylabel("Wartość miary")
        fig.legend(handle, ["Model trenowany na całym zbiorze trenującym"], loc='lower right')

        fig.tight_layout()
        
        # Save result
        fig.savefig(PLOTS_PATH / "all_metrics_comp" / str(dataset_classifier_combination_label + ".png"))

def plot_all_n_kcv_agg_one_metric_results(results: dict, metric: str):
    for dataset_classifier_combination_label, n_kcv_als_results in results.items():
        fig, ax = plt.subplots(1, 1, figsize=(10, 7), sharex=True, sharey=True)
        for n_kcv_al_result in n_kcv_als_results:
            al_method_name = list(n_kcv_al_result.keys())[0]
            al_method_result = list(n_kcv_al_result.values())[0]
            n_queries = len(al_method_result["al_classification"]["mean"][metric])
            x_axis = [(x/n_queries)*((1-INITIAL_TRAIN_SIZE)*100) + INITIAL_TRAIN_SIZE*100 for x in range(n_queries)]
            ax = plot_aggregated_metric(ax, x_axis, al_method_result["al_classification"]["mean"][metric], label=AL_LABEL_NAME_MAPPING[al_method_name])
        
        handle = plot_full_model_metric(ax, al_method_result["full_train_classification"]["mean"][metric], n_queries)

        ax.legend(loc='lower right')
        
        # Ticks setup
        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        
        dataset_name, classifier_name = dataset_classifier_combination_label.split('-')
        fig.suptitle(f"Zapytania aktywnego uczenia \n dla klasyfikatora {classifier_name} na zbiorze {dataset_name}. \n Metryka {METRIC_TITLE_NAME_MAPPING[metric]}", fontsize=16)
        fig.supxlabel("% zbioru trenującego")
        fig.supylabel("Wartość miary")
        fig.legend(handle, ["Model trenowany na całym zbiorze trenującym"], loc='lower right')

        fig.tight_layout()
        
        # Save result
        dir = PLOTS_PATH / f"{metric}_comp"
        try:
            os.makedirs(dir)
        except FileExistsError:
            pass
        fig.savefig(PLOTS_PATH / f"{metric}_comp" / str(dataset_classifier_combination_label + ".png"))