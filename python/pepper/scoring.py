from typing import List, Dict, Callable

from pepper.utils import print_subtitle, bold

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn import metrics
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    jaccard_score,
    accuracy_score
)


def get_similarity_score_fn_dict() -> Dict[str, Callable]:
    """
    Get a dictionary of similarity score functions for model evaluation.

    Returns
    -------
    Dict[str, Callable]
        A dictionary where keys are similarity score names and values are callable functions.
    """
    # For each of them, the best is 1 en the worst is 0
    return {
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "fbeta_5 (↑ recall)": lambda *args, **kwargs: fbeta_score(beta=5, *args, **kwargs),
        "fbeta_1_5 (↑ prec)": lambda *args, **kwargs: fbeta_score(beta=1/5, *args, **kwargs),
        "jaccard": jaccard_score,
    }


def global_similarity_report(
    cla_labels: List,
    clu_labels: List,
    indent: int = 0
) -> None:
    """
    Generate a global similarity report between classification labels and cluster labels.

    Parameters
    ----------
    cla_labels : List
        List of true classification labels.
    clu_labels : List
        List of cluster labels.
    indent : int, optional
        Indentation level for printing the report, by default 0.

    Returns
    -------
    None
    """
    print_subtitle("Global similarity report")

    score_funcs = get_similarity_score_fn_dict()  # Best is 1, worst is 0
    averages = ['micro', 'macro']  # Note: weighted not yet included

    # Simple base accuracy
    print(
        f"{(indent + 19) * ' '}accuracy: "
        f"{accuracy_score(cla_labels, clu_labels, normalize=False)}"
    )
    print(
        f"{(indent + 8) * ' '}normalized accuracy: "
        f"{accuracy_score(cla_labels, clu_labels):.2f}"
    )

    # Global scores (all classes are taken into account)
    for score_name, score_func in score_funcs.items():
        for avg in averages:
            score = score_func(cla_labels, clu_labels, average=avg)
            print(
                f"{(indent + 21 - len(score_name)) * ' '}"
                f"{score_name} {avg[:5]}: {score:.2f}"
            )


def local_similarity_report(
    cla_labels: List,
    clu_labels: List,
    indent: int = 0
) -> None:
    """
    Generate a local similarity report between classification labels and cluster labels.

    Parameters
    ----------
    cla_labels : List
        List of true classification labels.
    clu_labels : List
        List of cluster labels.
    indent : int, optional
        Indentation level for printing the report, by default 0.

    Returns
    -------
    None
    """
    print_subtitle("Local similarity report")
    
    score_funcs = get_similarity_score_fn_dict()  # Best is 1, worst is 0
    labels = list(set(cla_labels))

    print(
        f"{(indent + 18 - 8) * ' '}{bold('class_id')}: "
        f"{', '.join([f'C_{i:02d}' for i in range(len(labels))])}"
    )
    for score_name, score_func in score_funcs.items():
        # labels=[label] filters on a unique label
        scores = [
            score_func(cla_labels, clu_labels, labels=[label], average="macro")
            for label in labels
        ]
        print(f"{(indent + 18 - len(score_name)) * ' '}{score_name}: "
                f"{', '.join([f'{score:.2f}' for score in scores])}")


def show_confusion_matrix(
    cla_labels: List[int],
    clu_labels: List[int],
    cl_names: List[str]
) -> None:
    """
    Display confusion matrix with predicted and true class labels.

    Parameters
    ----------
    cla_labels : List[int]
        List of true class labels.
    clu_labels : List[int]
        List of predicted class labels.
    cl_names : List[str]
        List of class names (corresponding to indices increasing order).

    Returns
    -------
    None

    Note
    ----
    There is a SKL built in alternative :
    >>> from sklearn.metrics import ConfusionMatrixDisplay
    >>> fig, ax = plt.subplots(figsize=(10, 5))
    >>> ConfusionMatrixDisplay.from_predictions(
    >>>     cla_labels, aligned_clu_labels, ax=ax
    >>> )
    >>> ax.xaxis.set_ticklabels(cla_names)
    >>> ax.yaxis.set_ticklabels(cla_names)
    >>> _ = ax.set_title(f"Confusion Matrix")
    """
    # print_subtitle("Confusion matrix")
    conf_mx = metrics.confusion_matrix(cla_labels, clu_labels)
    cla_names = [
        cln[:13] + ("..." if len(cln) > 16 else cln[13:16])
        for cln in cl_names  # [P6] list(get_class_label_name_map().values())
    ]
    conf_data = pd.DataFrame(conf_mx, index=cla_names, columns=cla_names)

    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_data, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.title("Confusion matrix", fontsize=15, pad=15, weight="bold")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.show()


def display_classification_report(
    cla_labels: List[int],
    clu_labels: List[int],
    cl_names: List[str]
) -> None:
    """
    Display classification report with metrics for each class.

    Parameters
    ----------
    cla_labels : List[int]
        List of true class labels.
    clu_labels : List[int]
        List of predicted class labels.
    cl_names : List[str]
        List of class names (corresponding to indices increasing order).
    
    Returns
    -------
    None
    """
    #cla_names = cl_names  # [P6] list(get_class_label_name_map().values())
    print_subtitle("Classification report")
    print(metrics.classification_report(
        cla_labels, clu_labels, target_names=cl_names
    ))


def show_multilabel_confusion_matrixes(
    cla_labels: List[int],
    clu_labels: List[int],
    cl_names: List[str]
) -> None:
    """
    Display multiple confusion matrices for multilabel classification.

    Parameters
    ----------
    cla_labels : List[int]
        List of true class labels.
    clu_labels : List[int]
        List of predicted class labels.
    cl_names : List[str]
        List of class names (corresponding to indices increasing order).

    Returns
    -------
    None
    """
    def _plot_conf_mx(cfmx, cla_name, ax):
        sns.heatmap(cfmx, annot=True, fmt="d", linewidths=.5, ax=ax)
        ax.set_title(f"{cla_name} ConfMx")

    print_subtitle("multilabel confusion matrixes")
    labels = list(set(cla_labels))
    # P6 cla_names = list(get_class_label_name_map().values())
    conf_mx = metrics.multilabel_confusion_matrix(
        cla_labels, clu_labels, labels=labels
    )

    fig = plt.figure(figsize=(15, 7))
    for i in range(len(cl_names)):
        _plot_conf_mx(conf_mx[i], cl_names[i], ax=fig.add_subplot(240 + i + 1))
    plt.suptitle("Multilabel Confusion Matrixes", fontsize=15, weight="bold")
    plt.show()
