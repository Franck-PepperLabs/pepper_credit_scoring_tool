from typing import Dict, Union, Optional, Callable
import logging

import pandas as pd
import numpy as np

from pepper.metrics import require_probas

from pepper.debug import kv


def eval_predictions(
    eval_metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
    scores: Dict[str, Union[float, int]],
    eval_type: str,
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred_proba: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred_discr: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_true_name: Optional[str],
    eval_type_name: Optional[str],
    verbosity: Optional[int] = 0
) -> None:
    """
    Evaluate predictions using the given evaluation metrics.

    Parameters
    ----------
    eval_metrics : dict
        Dictionary of evaluation metrics to use. The keys are the names of the
        metrics and the values are callables that take two numpy arrays
        (`y_true` and `y_pred`) and return a float score.
    scores : dict
        Dictionary to store the evaluation scores in.
    eval_type : str
        Type of evaluation, e.g. "over_folds" or "overall".
    y_true : pandas.DataFrame, pandas.Series, or numpy.ndarray
        True labels for the evaluation data.
    y_pred_proba : pandas.DataFrame, pandas.Series, or numpy.ndarray
        Predicted class probabilities for the evaluation data.
    y_pred_discr : pandas.DataFrame, pandas.Series, or numpy.ndarray
        Predicted discrete class labels for the evaluation data.
    y_true_name : str, optional
        Name of the `y_true` array (used for display purposes only).
    eval_type_name : str, optional
        Name of the `eval_type` parameter (used for display purposes only).
    verbosity : int, optional
        Verbosity level. If 0, no messages are printed.
        If > 0, progress messages are printed.

    Notes
    -----
    For discrete evaluation metrics, the discrete prediction must be used.
    For metrics that accept probabilistic predictions, using the predicted
    class probabilities may give better results.

    Examples
    --------
    >>> eval_predictions(
    ...     eval_metrics, scores, "over_folds",
    ...     y_valid, y_pred_proba, y_pred_discr,
    ...     y_true_name="y_valid", eval_type_name=f"Fold {fold_id:2d}",
    ...     verbosity=verbosity
    ... )

    >>> eval_predictions(
    ...     eval_metrics, scores, "overall",
    ...     y_train, oof_preds_proba, oof_preds_discr,
    ...     y_true_name="y_train", eval_type_name=f"Full",
    ...     verbosity=verbosity
    ... )
    """
    # Avoid ValueError:
    #   Only one class present in y_true. The scores are not defined in that case.
    if y_true.nunique() > 1:
        for name, eval in eval_metrics.items():
            y_pred = y_pred_proba if require_probas(eval) else y_pred_discr
            score = eval(y_true, y_pred)
            if eval_type == "over_folds":
                scores[name][eval_type].append(score)
            else:  # eval_type == "overall":
                scores[name][eval_type] = score
            if verbosity > 0:
                kv(2, f"{eval_type_name} {name}", f"{score:.6f}")
    elif verbosity > 0:
        print(
            f"Only one class present in `{y_true_name}`. "
            "The scores are not defined in that case."
        )


def log_eval_predictions_v2(
    eval_metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_discr: np.ndarray,
    y_true_name: str = "y_true",
    eval_type_name: str = "Evaluation"
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Evaluate predictions using the given evaluation metrics.

    Parameters
    ----------
    eval_metrics : dict
        Dictionary of evaluation metrics to use. The keys are the names of the
        metrics, and the values are callables that take two numpy arrays
        (`y_true` and `y_pred`) and return a float score.

    y_true : numpy.ndarray
        True labels for the evaluation data.

    y_pred_proba : numpy.ndarray
        Predicted class probabilities for the evaluation data.

    y_pred_discr : numpy.ndarray
        Predicted discrete class labels for the evaluation data.

    y_true_name : str, optional
        Name of the `y_true` array (used for display purposes only).

    eval_type_name : str, optional
        Name of the `eval_type` parameter (used for display purposes only).

    Returns
    -------
    dict
        A dictionary containing evaluation scores for each metric.
        The keys are the metric names, and the values are either float scores
        (if `eval_type` is "overall") or dictionaries of scores per fold
        (if `eval_type` is "over_folds").

    Notes
    -----
    For discrete evaluation metrics, the discrete prediction must be used.
    For metrics that accept probabilistic predictions, using the predicted
    class probabilities may give better results.
    """
    scores = {}
    
    # Avoid ValueError if only one class present in y_true
    if y_true.nunique() <= 1:
        logging.warning(
            f"Only one class present in `{y_true_name}`. "
            "The scores are not defined in that case."
        )
        return scores

    for metric_name, metric_func in eval_metrics.items():
        y_pred = y_pred_proba if require_probas(metric_func) else y_pred_discr
        score = metric_func(y_true, y_pred)
        scores[metric_name] = score
        logging.info(f"{eval_type_name} {metric_name}: {score:.6f}")

    return scores
