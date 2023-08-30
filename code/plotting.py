import uuid, os
import numpy as np
import matplotlib.pyplot as plt
from config import RANDOM_STATE_SEED, PLOTS_PATH

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

def plot_metrics(performance_history):
    # Plotting metrics
    fig, axs = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True)

    if type(full_model_score) is not list:
        full_model_score = list(full_model_score.values())
    metrics = list(zip(*[x.values() for x in performance_history]))

    for i, metric_name in enumerate(performance_history[0].keys()):
        axs[int(i/3), int(i%3)].plot(range(len(metrics[i])), metrics[i])
        handle = axs[int(i/3), int(i%3)].plot(range(len(metrics[i])), np.ones(len(metrics[i]))*full_model_score[i], "r")
        axs[int(i/3), int(i%3)].set_title(metric_name)

    fig.suptitle("Pool based learning metrics", fontsize=16)
    fig.supxlabel("Number of queries")
    fig.supylabel("Metric value")
    fig.legend(handle, ["Model trained on whole dataset"], loc='lower right')
    fig.tight_layout()


# confusion matrix
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