
from typing import Callable

from sklearn import metrics
""".metrics import (
    roc_auc_score, brier_score_loss, average_precision_score,
    precision_recall_curve, fbeta_score, roc_curve, confusion_matrix
)"""


def require_probas(metric: Callable) -> bool:
    """
    Check if a metric requires probabilistic predictions.

    Parameters
    ----------
    metric : callable
        Metric function.

    Returns
    -------
    bool
        True if the metric requires probabilistic predictions, False otherwise.
    """
    return metric in [
        metrics.roc_auc_score,
        metrics.brier_score_loss,
        metrics.average_precision_score,
        metrics.precision_recall_curve,
        metrics.log_loss,  # Log Loss
        metrics.f1_score,  # F1 Score
        metrics.log_loss   # Logistic Loss
    ]