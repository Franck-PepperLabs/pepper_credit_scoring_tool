from typing import Tuple, List, Dict, Any, Union, Optional, Callable
from collections import Counter

from IPython.display import display

import pandas as pd
import numpy as np

from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from imblearn.base import BaseSampler
from imblearn.combine import SMOTETomek
from imblearn import under_sampling

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn import metrics
""".metrics import (
    roc_auc_score, brier_score_loss, average_precision_score,
    precision_recall_curve, fbeta_score, roc_curve, confusion_matrix
)"""

import matplotlib.pyplot as plt
import seaborn as sns

# from home_credit.lightgbm_kernel_v2 import display_importances
# from home_credit.load import save_submission
from pepper.debug import kv, tx, tl, stl
from pepper.utils import save_and_show, display_key_val

from home_credit.load import load_prep_dataset
from home_credit.impute import default_imputation

from home_credit.model.preprocessing import train_preproc
from home_credit.model.eval import eval_predictions


# Eval a model with KFold or Stratified KFold
def kfold_train_and_eval_model(
    data: pd.DataFrame,
    clf: ClassifierMixin,
    imb_sampler: Union[BaseSampler, None] = SMOTETomek(),
    scaler: Union[TransformerMixin, None] = MinMaxScaler(),
    loss_func: Union[
        Dict[str, Callable[[np.ndarray, np.ndarray], float]],
        None] = {"AUC": metrics.roc_auc_score},
    eval_metrics: Union[
        Dict[str, Callable[[np.ndarray, np.ndarray], float]],
        None] = {
        # "AUC": roc_auc_score,
        "F2": lambda y, y_pred: metrics.fbeta_score(y, y_pred, beta=2)
    },
    nfolds: int = 5,
    stratified: bool = False,
    verbosity: int = 0,
    return_trained_clf: bool = False,
) -> Dict[str, Union[np.ndarray, Dict[str, Union[float, int]], Any]]:
    """Trains and evaluates a classifier using k-fold cross-validation.

    Parameters
    ----------
    data : pd.DataFrame
        Input data to be used for training and evaluation. It should contain
        the target variable and features to be used for training the model.
    clf : ClassifierMixin
        Classifier to be used for training and evaluation.
    imb_sampler : Union[BaseSampler, None], optional
        Imbalanced dataset sampler to be used for oversampling the minority
        class during training. Defaults to SMOTETomek().
        The sampling is deactivated if None.
    scaler : Union[TransformerMixin, None], optional
        Feature scaler to be used during training. Defaults to MinMaxScaler().
        The scaling is deactivated if None.
    loss_func : Union[Callable[[np.ndarray, np.ndarray], None], optional
        The loss function to be used for optimization during training.
        Defaults to {"AUC": roc_auc_score}
    eval_metrics : Dict[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Dictionary of evaluation metrics to be used for evaluating the model.
        The keys should be strings indicating the name of the metric, and the
        values should be callables that take two arrays (the true and predicted
        labels) and return a float.
        The given `eval_metrics` is completed with the `loss_func`.
        Defaults to {
            "F2": lambda y_true, y_pred: fbeta_score(y_true, y_pred, beta=2)
        }
    nfolds : int, optional
        Number of folds to be used for cross-validation. Defaults to 5.
    stratified : bool, optional
        Whether to use stratified k-fold cross-validation or not.
        Defaults to False.
    verbosity : int, optional
        Verbosity level of the function. The higher the level, the more
        information will be printed during training and evaluation.
        Defaults to 0 (silent).
    return_trained_clf : bool, optional
        Whether to return the trained classifier in the output or not.
        Defaults to False.

    Returns
    -------
    results : Dict[str, Union[np.ndarray, Dict[str, Union[float, int]], Any]]
        A dictionary containing the test predictions, feature importances,
        and performance scores of the trained model. The keys are "test_preds"
        (np.ndarray), "feature_importance" (Dict[str, Union[float, int]]),
        and "scores" (Dict[str, float]).
        If `return_trained_clf` is True, the dictionary will also contain a
        "trained_clf" key with the trained classifier object.
    """
    if loss_func is None:
        loss_func = {"AUC": metrics.roc_auc_score}

    if eval_metrics is None:
        eval_metrics = {}

    eval_metrics.update(loss_func)

    # Default imputation, if necessary
    # (should have been handled by feature engineering)
    if (data.isna() | np.isinf(data)).any().any():
        data = default_imputation(data)

    # Exclude non-feature columns from training and test features
    not_feat_names = ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
    feat_names = data.columns.difference(not_feat_names)
    X = data[feat_names]
    y = data.TARGET

    # Print the shape of the training and test data
    tl(verbosity, "Starting train and eval of:")
    kv(verbosity, "Labeled dataset of shape", f"{data.shape}")
    kv(verbosity, "Features set of shape", f"{X.shape}")
    tl(verbosity, "With:")
    kv(verbosity, "Classifier", ""), tx(verbosity, clf) # TODO : kv with display arg
    kv(verbosity, "Loss function", ""), tx(verbosity, loss_func)
    kv(verbosity, "Eval metrics", ""), tx(verbosity, eval_metrics)
    kv(verbosity,
        f"On {nfolds} {'stratifed ' if stratified else ''}KFolds", ""
    )

    # Resample the data in a balanced set
    if imb_sampler is not None:
        X_res, y_res = imb_sampler.fit_resample(X, y)
        stl(verbosity, "Resampling")
        kv(verbosity, "Sampler", ""), tx(verbosity, imb_sampler) # TODO : kv with display arg
        kv(verbosity, "Original dataset shape", f"{Counter(y)}")
        kv(verbosity, "Resampled dataset shape", f"{Counter(y_res)}")
    else:
        X_res, y_res = X, y
        stl(verbosity, "Warning: The target classes are imbalanced!")
        tx(verbosity,
            "You should consider to pass a `imblearn` resampler "
            "through the `imb_sampler` parameter.")

    # Split the target variable and the features for training and test data
    X_train = X_res[y_res > -1].copy()
    y_train = y_res[y_res > -1].copy()
    X_test = X_res[y_res == -1].copy()

    # Print the shape of the training and test data
    stl(verbosity, "Train vs. test subsets shapes")
    kv(verbosity, "\tTrain shape", f"{X_train.shape}")
    kv(verbosity, "\tTest shape", f"{X_test.shape}")

    # Scale the data
    if scaler is not None:
        scaler.fit(X_train)
        X_train = pd.DataFrame(
            scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

    # Create the cross-validation model
    fold_params = {"n_splits": nfolds, "shuffle": True, "random_state": 42}
    folds = (StratifiedKFold if stratified else KFold)(**fold_params)

    # Create arrays and dataframes to store results
    # `oof_preds` will store the out-of-fold predictions for the training data
    # `sms_preds` will store the submission predictions for the test data 
    oof_preds_proba = np.zeros(X_train.shape[0])
    oof_preds_discr = np.zeros(X_train.shape[0])
    test_pred_proba = np.zeros(X_test.shape[0])

    # Iterate through the folds
    fold_imps = []
    scores = {score_name: {"over_folds": []} for score_name in eval_metrics.keys()}
    for fold_id, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
        stl(verbosity, f"Evaluate the {fold_id+1}-th fold (on {nfolds})")
        # Split the training and validation data using the current fold
        X_y_split = lambda idx: (X_train.iloc[idx], y_train.iloc[idx])
        X_y_train = X_y_split(train_idx)
        X_y_valid = X_y_split(valid_idx)
        X_valid, y_valid = X_y_valid

        # Train the classifier using the training and validation data and the business loss function
        fit_facade(clf, X_y_train, X_y_valid, loss_func)

        # Make predictions on the validation and test data using the trained model
        y_pred_proba = oof_preds_proba[valid_idx] = predict_proba_facade(clf, X_valid)
        y_pred_discr = oof_preds_discr[valid_idx] = predict_facade(clf, X_valid)

        # Aggregate the submission predictions (test predictions) for the current fold
        test_pred_proba += predict_proba_facade(clf, X_test) / nfolds

        # Get the feature importances for the current fold
        fold_imp = pd.DataFrame({
            "feature": feat_names,
            "importance": get_feature_importances_facade(clf),   # clf.feature_importances_,
            "fold": fold_id
        })

        # Concatenate the feature importances across all folds
        fold_imps.append(fold_imp)

        # ? déjà fait plus haut ? y_pred_discr = clf.predict(X_valid)

        # Compute the scores for the current fold
        eval_predictions(
            eval_metrics, scores, "over_folds",
            y_valid, y_pred_proba, y_pred_discr,
            "y_valid", f"Fold {fold_id:2d}", verbosity
        )    

    # Compute the overall train scores
    eval_predictions(
        eval_metrics,
        scores,
        "overall",
        y_train,
        oof_preds_proba,
        oof_preds_discr,
        "y_train",
        "Full",
        verbosity,
    )

    # Concatenate the feature importances across all folds
    feat_imp = pd.concat(fold_imps, axis=0)

    # Return :
    #   resampled train, test sets and train target
    #   train and test probabilistic and discrete predictions,
    #   performance scores and feature importances
    res = {
        "resamples": {"X_train": X_train, "X_test": X_test,"y_train": y_train},
        "preds": {
            "train": {"proba": oof_preds_proba, "discr": oof_preds_discr},
            "test": {"proba": test_pred_proba, "discr": np.round(test_pred_proba)}
        },
        "scores": scores, "feat_imps": feat_imp
    }

    # Completed with the trained classifier if asked (optional)
    if return_trained_clf:
        res["trained_clf"] = clf

    return res



# Eval a model with KFold or Stratified KFold
def kfold_train_and_eval_model_v2(
    data: pd.DataFrame,
    clf: ClassifierMixin,
    imb_sampler: Union[BaseSampler, None] = under_sampling.RandomUnderSampler(random_state=42),
    scaler: Union[TransformerMixin, None] = MinMaxScaler(),
    loss_func: Union[
        Dict[str, Callable[[np.ndarray, np.ndarray], float]],
        None] = {"AUC": metrics.roc_auc_score},
    eval_metrics: Union[
        Dict[str, Callable[[np.ndarray, np.ndarray], float]],
        None] = {
        # "AUC": roc_auc_score,
        "F2": lambda y, y_pred: metrics.fbeta_score(y, y_pred, beta=2)
    },
    nfolds: int = 5,
    stratified: bool = False,
    verbosity: int = 0,
    return_trained_clf: bool = False,
) -> Dict[str, Union[np.ndarray, Dict[str, Union[float, int]], Any]]:
    """Trains and evaluates a classifier using k-fold cross-validation.

    Parameters
    ----------
    data : pd.DataFrame
        Input data to be used for training and evaluation. It should contain
        the target variable and features to be used for training the model.
    clf : ClassifierMixin
        Classifier to be used for training and evaluation.
    imb_sampler : Union[BaseSampler, None], optional
        Imbalanced dataset sampler to be used for oversampling the minority
        class during training. Defaults to SMOTETomek().
        The sampling is deactivated if None.
    scaler : Union[TransformerMixin, None], optional
        Feature scaler to be used during training. Defaults to MinMaxScaler().
        The scaling is deactivated if None.
    loss_func : Union[Callable[[np.ndarray, np.ndarray], None], optional
        The loss function to be used for optimization during training.
        Defaults to {"AUC": roc_auc_score}
    eval_metrics : Dict[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Dictionary of evaluation metrics to be used for evaluating the model.
        The keys should be strings indicating the name of the metric, and the
        values should be callables that take two arrays (the true and predicted
        labels) and return a float.
        The given `eval_metrics` is completed with the `loss_func`.
        Defaults to {
            "F2": lambda y_true, y_pred: fbeta_score(y_true, y_pred, beta=2)
        }
    nfolds : int, optional
        Number of folds to be used for cross-validation. Defaults to 5.
    stratified : bool, optional
        Whether to use stratified k-fold cross-validation or not.
        Defaults to False.
    verbosity : int, optional
        Verbosity level of the function. The higher the level, the more
        information will be printed during training and evaluation.
        Defaults to 0 (silent).
    return_trained_clf : bool, optional
        Whether to return the trained classifier in the output or not.
        Defaults to False.

    Returns
    -------
    results : Dict[str, Union[np.ndarray, Dict[str, Union[float, int]], Any]]
        A dictionary containing the test predictions, feature importances,
        and performance scores of the trained model. The keys are "test_preds"
        (np.ndarray), "feature_importance" (Dict[str, Union[float, int]]),
        and "scores" (Dict[str, float]).
        If `return_trained_clf` is True, the dictionary will also contain a
        "trained_clf" key with the trained classifier object.
    """
    if loss_func is None:
        loss_func = {"AUC": metrics.roc_auc_score}

    if eval_metrics is None:
        eval_metrics = {}
    eval_metrics.update(loss_func)

    """
    # Default imputation, if necessary
    # (should have been handled by feature enginering)
    if (data.isna() | np.isinf(data)).any().any():
        data = default_imputation(data)

    # Exclude non-feature columns from training and test features
    not_feat_names = ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
    feat_names = data.columns.difference(not_feat_names)
    X = data[feat_names]
    y = data.TARGET
    """
    X, y = train_preproc(data, scaler)

    # Print the shape of the training and test data
    tl(verbosity, "Starting train and eval of:")
    kv(verbosity, "Labeled dataset of shape", f"{data.shape}")
    kv(verbosity, "Features set of shape", f"{X.shape}")
    tl(verbosity, "With:")
    kv(verbosity, "Classifier", ""), tx(verbosity, clf) # TODO : kv with display arg
    kv(verbosity, "Loss function", ""), tx(verbosity, loss_func)
    kv(verbosity, "Eval metrics", ""), tx(verbosity, eval_metrics)
    kv(verbosity,
        f"On {nfolds} {'stratified ' if stratified else ''}KFolds", ""
    )

    # Resample the data in a balanced set
    if imb_sampler is not None:
        X_res, y_res = imb_sampler.fit_resample(X, y)
        stl(verbosity, "Resampling")
        kv(verbosity, "Sampler", ""), tx(verbosity, imb_sampler) # TODO : kv with display arg
        kv(verbosity, "Original dataset shape", f"{Counter(y)}")
        kv(verbosity, "Resampled dataset shape", f"{Counter(y_res)}")
    else:
        X_res, y_res = X, y
        stl(verbosity, "Warning: The target classes are imbalanced!")
        tx(verbosity,
            "You should consider to pass a `imblearn` resampler "
            "through the `imb_sampler` parameter.")

    # Split the target variable and the features data
    X = X_res[y_res > -1].copy()
    y = y_res[y_res > -1].copy()

    # Print the shape of the training data
    kv(verbosity, "Data shape", f"{X.shape}")

    # Scale the data
    """if scaler is not None:
        scaler.fit(X)
        X = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )"""

    # Create the cross-validation model
    fold_params = {"n_splits": nfolds, "shuffle": True, "random_state": 42}
    folds = (StratifiedKFold if stratified else KFold)(**fold_params)

    # Create arrays and dataframes to store results
    # `oof_preds` will store the out-of-fold predictions for the training data
    # `sms_preds` will store the submission predictions for the test data 
    oof_preds_proba = np.zeros(X.shape[0])
    oof_preds_discr = np.zeros(X.shape[0])
    #test_pred_proba = np.zeros(X_test.shape[0])

    # Iterate through the folds
    fold_imps = []
    scores = {score_name: {"over_folds": []} for score_name in eval_metrics.keys()}
    for fold_id, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        stl(verbosity, f"Evaluate the {fold_id+1}-th fold (on {nfolds})")
        # Split the training and validation data using the current fold
        X_y_split = lambda idx: (X.iloc[idx], y.iloc[idx])
        X_y_train = X_y_split(train_idx)
        X_y_valid = X_y_split(valid_idx)
        X_valid, y_valid = X_y_valid

        # Train the classifier using the training and validation data and the business loss function
        fit_facade(clf, X_y_train, X_y_valid, loss_func)

        # Make predictions on the validation and test data using the trained model
        y_pred_proba = oof_preds_proba[valid_idx] = predict_proba_facade(clf, X_valid)
        y_pred_discr = oof_preds_discr[valid_idx] = predict_facade(clf, X_valid)

        # Get the feature importances for the current fold
        fold_imp = pd.DataFrame({
            "feature": X.columns, # feat_names,
            "importance": get_feature_importances_facade(clf),   # clf.feature_importances_,
            "fold": fold_id
        })

        # Concatenate the feature importances across all folds
        fold_imps.append(fold_imp)

        # ? déjà fait plus haut ? y_pred_discr = clf.predict(X_valid)

        # Compute the scores for the current fold
        eval_predictions(
            eval_metrics, scores, "over_folds",
            y_valid, y_pred_proba, y_pred_discr,
            "y_valid", f"Fold {fold_id:2d}", verbosity
        )    

    # Compute the overall train scores
    eval_predictions(
        eval_metrics,
        scores,
        "overall",
        y,
        oof_preds_proba,
        oof_preds_discr,
        "y_train",
        "Full",
        verbosity,
    )

    # Concatenate the feature importances across all folds
    feat_imp = pd.concat(fold_imps, axis=0)

    # Return :
    #   resampled data features and target
    #   probabilistic and discrete predictions (on oof valid data),
    #   performance scores and feature importances
    res = {
        "resamples": {"X": X,"y": y},
        "preds": {"proba": oof_preds_proba, "discr": oof_preds_discr},
        "scores": scores, "feat_imps": feat_imp
    }

    # Completed with the trained classifier if asked (optional)
    if return_trained_clf:
        res["trained_clf"] = clf

    return res


""" Predict
"""

def predict(
    clf: ClassifierMixin, 
    X: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Predict the class probabilities and the binary classification labels
    of a classifier for a given set of input features X using a specified 
    probability threshold.

    Parameters
    ----------
    clf : ClassifierMixin
        The trained classifier object.
    X : np.ndarray
        The input features for prediction.
    threshold : float
        The threshold probability for binary classification.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two 1-dimensional numpy arrays:
        * y_pred : The binary classification labels predicted by the classifier.
        * y_pred_proba : The predicted probability of the positive class.
    """
    y_pred_proba = clf.predict_proba(X)[:, 1]
    y_pred = np.where(y_pred_proba > threshold, 1, 0)
    return y_pred, y_pred_proba


""" Display and plot evals
"""


def display_train_and_eval_results(res):
    # DEPRECATED ?
    X_train_res = res["resamples"]["X"]
    y_train_res = res["resamples"]["y"]
    print(f"resampled X : shape {X_train_res.shape}")
    print(f"classe counts {np.unique(y_train_res, return_counts=True)}")

    y_train_res_pred_proba = res["preds"]["proba"]
    y_train_res_pred_discr = res["preds"]["discr"]
    print("y_train_res_pred_proba:", y_train_res_pred_proba)
    print("y_train_res_pred_discr:", y_train_res_pred_discr)
    print(f"preds counts {np.unique(y_train_res_pred_discr, return_counts=True)}")

    train_res_scores = res["scores"]
    display(train_res_scores)

    # train_res_feat_imps = res["feat_imps"]



def evaluate_classifier(
    clf: ClassifierMixin, 
    X: np.ndarray, 
    y_true: np.ndarray, 
    threshold: float
) -> None:
    """
    Evaluate the performance of a binary classifier.

    Calculates the following metrics:
    * ROC AUC
    * Accuracy
    * Adjusted Rand Index
    * Jaccard Index
    * Precision
    * Recall
    * F-beta scores, with beta set to 2^k, k = 0, ..., 4,
        aiming to emphasize precision over recall

    Parameters
    ----------
    clf : ClassifierMixin
        A trained binary classifier object with a predict_proba() method.
    X : numpy.ndarray
        The input data of shape (n_samples, n_features).
    y_true : numpy.ndarray
        The true binary labels of shape (n_samples,).
    threshold : float
        A threshold value to apply on predicted probabilities to obtain binary labels.

    Returns
    -------
    None
    """
    # Get the predictions
    y_pred, y_pred_proba = predict(clf, X, threshold)
    # Diplay the macro scores
    display_key_val("ROC AUC", metrics.roc_auc_score(y_true, y_pred_proba))
    display_key_val("Accuracy", metrics.accuracy_score(y_true, y_pred))
    display_key_val("ARI", metrics.adjusted_rand_score(y_true, y_pred))
    # Display the micro scores
    kwargs = {"y_true": y_true, "y_pred": y_pred, "average": "micro"}
    display_key_val("Jaccard Index", metrics.jaccard_score(**kwargs))
    display_key_val("Precision", metrics.precision_score(**kwargs))
    display_key_val("Recall", metrics.recall_score(**kwargs))
    display_key_val("F1", metrics.f1_score(**kwargs))
    display_key_val("F2", metrics.fbeta_score(**kwargs, beta=2))
    display_key_val("F4", metrics.fbeta_score(**kwargs, beta=4))
    display_key_val("F8", metrics.fbeta_score(**kwargs, beta=8))
    display_key_val("F16", metrics.fbeta_score(**kwargs, beta=16))


# Adpated from `pepper.scoring.show_confusion_matrix`
def ax_plot_confusion_matrix(
    cla_labels: List[int],
    clu_labels: List[int],
    cl_names: List[str],
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Union[str, None]:
    """Displays a confusion matrix with predicted and true class labels.

    Parameters
    ----------
    cla_labels : List[int]
        List of true class labels.
    clu_labels : List[int]
        List of predicted class labels.
    cl_names : List[str]
        List of class names (corresponding to indices increasing order).
    title : Optional[str], default=None
        Title for the plot.
    ax : Optional[plt.Axes], default=None
        Matplotlib axes to plot on.
        If not provided, a new figure will be created.

    Returns
    -------
    Union[str, None]

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
    conf_mx = metrics.confusion_matrix(cla_labels, clu_labels)
    cla_names = [
        cln[:13] + ("..." if len(cln) > 16 else cln[13:16])
        for cln in cl_names
    ]
    conf_data = pd.DataFrame(conf_mx, index=cla_names, columns=cla_names)

    # Draw a heatmap with the numeric values in each cell
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax = sns.heatmap(conf_data, annot=True, fmt="d", linewidths=.5, ax=ax)
    if title is not None:
        ax.set_title("Confusion matrix", fontsize=15, pad=15)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    # Adjust the spacing between the subplots and save/show the figure
    if fig is not None:
        fig.tight_layout()
        if title is None:
            title = "Jane DOE"
        return save_and_show(
            f"{title.lower()}",
            sub_dir="model_eval",
            return_filepath=True
        )



def plot_roc_curve(
    y_true: Union[List[int], np.ndarray],
    y_pred_proba: Union[List[float], np.ndarray],
    overall_auc: float,
    title: Optional[str] = None
) -> str:
    """DEPRECATED: Use `ax_plot_roc_curve` instead.

    Display ROC curve for binary classification model.

    Parameters
    ----------
    y_true : Union[List[int], np.ndarray]
        True binary labels (0 or 1).
    y_pred_proba : Union[List[float], np.ndarray]
        Predicted probabilities for positive class (label 1).
    overall_auc : float
        Area under the ROC curve (AUC) score.
    title : Optional[str], default=None
        Title of the plot.

    Returns
    -------
    None
    """
    fpr, tpr,_ = metrics.roc_curve(y_true, y_pred_proba)
    plt.plot(
        fpr, tpr,
        label=f"Model AUC = {overall_auc:0.4f}",
        color="magenta"
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle=':',
        label="Baseline AUC = 0.5",
        color="yellow",
    )
    plt.fill_between(fpr, fpr, tpr, color='magenta', alpha=0.2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    # Add a title to the figure
    if title is None:
        title = "ROC AUC Curve\n"
    else:
        title = f"ROC AUC Curve for `{title}`\n"
    plt.title(title, fontsize=15, weight="bold", pad=15)

    # Adjust the spacing between the subplots and save/show the figure
    plt.tight_layout()
    return save_and_show(f"{title.lower()}", sub_dir="model_eval", return_filepath=True)



def ax_plot_roc_curve(
    y_true: Union[List[int], np.ndarray],
    y_pred_proba: Union[List[float], np.ndarray],
    overall_auc: float,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (4, 3)
) -> Union[str, None]:
    """
    Display ROC curve for binary classification model.

    Parameters
    ----------
    y_true : Union[List[int], np.ndarray]
        True binary labels (0 or 1).
    y_pred_proba : Union[List[float], np.ndarray]
        Predicted probabilities for positive class (label 1).
    overall_auc : float
        Area under the ROC curve (AUC) score.
    title : Optional[str], default=None
        Title of the plot.
    ax : Optional[plt.Axes], default=None
        Matplotlib Axes object to plot on.
        If not provided, a new figure and axes will be created.
    figsize : Tuple[float, float], default=(4, 3)
        Figure size (width, height) in inches.

    Returns
    -------
    Union[str, None]
    """
    fpr, tpr,_ = metrics.roc_curve(y_true, y_pred_proba)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        fpr, tpr,
        label=f"Model AUC = {overall_auc:0.4f}",
        color="magenta"
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle=':',
        label="Baseline AUC = 0.5",
        color="yellow",
    )
    ax.fill_between(fpr, fpr, tpr, color='magenta', alpha=0.2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()

    # Add a title to the figure
    title = "\n" if title is None else f"{title}\n"
    ax.set_title(title, fontsize=15, pad=15)

    # Adjust the spacing between the subplots and save/show the figure
    if fig is not None:
        fig.tight_layout()
        return save_and_show(
            f"{title.lower()}",
            sub_dir="model_eval",
            return_filepath=True
        )


def display_subset_info(
    X: np.ndarray, 
    y: np.ndarray, 
    _ind: str = ""
) -> None:
    """
    Print information about a subset of data.

    Parameters
    ----------
    X : np.ndarray
        The features of the subset.
    y : np.ndarray
        The target labels of the subset.
    _ind : str, optional
        A suffix to add to the feature and label names. Default is "".

    Returns
    -------
    None
    """
    print(
        f"X{_ind} : shape {X.shape}"
        f"\ty{_ind} class counts : {np.unique(y, return_counts=True)}"
    )


def plot_model_eval_roc_and_confusion(
    inputss: List[Tuple[List[int], List[int], List[float], float, str]],
    single_figsize: Tuple[int, int] = (5, 5),
    title: str = None
) -> str:
    """
    Plot the ROC AUC curves and confusion matrices for multiple models
    in a single figure.

    Parameters
    ----------
    inputss: List[Tuple[List[int], List[int], List[float], float, str]]
        A list of tuples containing the inputs for each model to plot.
        Each tuple should have the following elements:
        - y: list of true binary labels (0s and 1s)
        - y_pred: list of predicted binary labels (0s and 1s)
        - y_pred_proba:
            list of predicted probabilities for the positive class (1s)
        - auc: the ROC AUC score for the model
        - ttl: the title for the subplot of the model
    single_figsize: Tuple[int, int]
        The size of each individual subplot, as a tuple of (width, height)
        in inches. Default is (5, 5).
    title: str
        The title for the entire figure.
        Default is "ROC AUC and Confusion Mx\n".

    Returns:
    --------
    The absolute path of the file where the graph image has been saved.
    """
    # Create the figure and subplots
    nrows = 2
    ncols = 4
    w, h = single_figsize
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(w * ncols, h * nrows)
    )

    # Plot the ROC AUCs
    for inputs, ax_roc, ax_conf in zip(inputss, axes[0], axes[1]):
        y, y_pred, y_pred_proba, auc, ttl = inputs
        ax_plot_roc_curve(y, y_pred_proba, auc, ttl, ax=ax_roc)
        ax_plot_confusion_matrix(y, y_pred, ["Repaid", "Past due"], ax=ax_conf)

    # Add a title to the figure
    if title is None:
        title = "ROC AUC and Confusion Mx\n"
    else:
        title = f"ROC AUC and Confusion Mx for `{title}`\n"
    plt.suptitle(title, fontsize=15, weight="bold")

    # Adjust the spacing between the subplots and save/show the figure
    fig.tight_layout()
    return save_and_show(
        f"{title.lower()}",
        sub_dir="model_eval",
        return_filepath=True
    )


def post_train_eval(
    data_s: pd.DataFrame,
    res: Dict[str, Union[np.ndarray, Dict[str, Union[float, int]], Any]],
    clf: ClassifierMixin,
    scaler: Union[TransformerMixin, None] = MinMaxScaler(),
    threshold: float = .5
) -> str:
    """
    Evaluate the performance of the trained classifier on different subsets
    of the dataset :
    * (X, y) : the entire training dataset
    * (X_s, y_s) : the subset used for training
    * (X_rs, y_rs) : the resampled training dataset
    * (X_v, y_v) : the validation dataset
        (i.e., the part of the training dataset not used for training)

    Parameters
    ----------
    data_s : pandas.DataFrame
        A subset of the original training dataset.
    res : dict
        Results of resampling and predictions made on the resampled dataset.
    clf : object
        A classifier object trained on the preprocessed data.
    scaler : object
        A scaler object used to preprocess the data.
    threshold : float
        A threshold value to apply on predicted probabilities to obtain binary labels.
    Returns
    -------
    The absolute path of the file where the graph image has been saved.
    """
    tl(3, f"Post train eval with threshold of {100*threshold:.1f} %")

    # Load the preprocessed dataset
    data = load_prep_dataset("train_baseline_all")

    # Preprocess the entire training dataset
    X, y = train_preproc(data, scaler)
    display_subset_info(X, y, "")
    
    # Get the preprocessed subset used for training
    s_index = data_s.index
    X_s, y_s = X.loc[s_index], y.loc[s_index]
    display_subset_info(X_s, y_s, "s")

    # Get the resampled preprocessed training dataset
    # TODO : simplement retourner l'index -> modifier la sortie de l'entraineur
    X_rs, y_rs = res["resamples"]["X"], res["resamples"]["y"]
    display_subset_info(X_rs, y_rs, "rs")
    
    # Get the preprocessed validation dataset
    # (i.e., the part of the training dataset not used for training)
    v_index = ~data.index.isin(data_s.index)
    X_v, y_v = X[v_index], y[v_index]
    display_subset_info(X_v, y_v, "v")

    # Evaluate the performance of the resampled training dataset
    y_rs_pred_proba = res["preds"]["proba"]
    y_rs_pred = res["preds"]["discr"]
    auc_rs = res["scores"]['AUC']['overall']

    # Evaluate the performance of the training subset
    y_s_pred, y_s_pred_proba = predict(clf, X_s, threshold)
    auc_s = metrics.roc_auc_score(y_s, y_s_pred_proba)

    # Evaluate the performance of the entire training dataset
    y_pred, y_pred_proba = predict(clf, X, threshold)
    auc = metrics.roc_auc_score(y, y_pred_proba)

    # Evaluate the performance of the validation subset
    y_v_pred, y_v_pred_proba = predict(clf, X_v, threshold)
    auc_v = metrics.roc_auc_score(y_v, y_v_pred_proba)

    # Plot the evaluation metrics for all subsets
    main_image_filepath = plot_model_eval_roc_and_confusion(
        [
            (y_rs, y_rs_pred, y_rs_pred_proba, auc_rs, "Resampled train subset"),
            (y_s, y_s_pred, y_s_pred_proba, auc_s, "Training subset"),
            (y, y_pred, y_pred_proba, auc, "Full train set"),
            (y_v, y_v_pred, y_v_pred_proba, auc_v, r"Validation subset (full\training)")
        ],
        title=f"Sample 10k (thres: {100*threshold:.1f})"
    )

    # Terminons par les mesures classiques pour la classification avec du f beta de 2, 4, 8, 16
    # stl(v, "Compute scores")
    # sh_t = -timeit.default_timer()
    # réintroduire le ctx_mgr ?
    stl(3, "Validation set metrics")
    evaluate_classifier(clf, X_v, y_v, threshold=threshold)

    # Les mêmes mesure qui sont déséquilibrées, après utilisation de SMOTETomek
    # Resample the validation data in a balanced set
    stl(3, "Undersampled validation set metrics")
    imb_sampler = under_sampling.RandomUnderSampler(random_state=42)
    X_rv, y_rv = imb_sampler.fit_resample(X_v, y_v)
    print("Original dataset shape", f"{Counter(y_v)}")
    print("Resampled dataset shape", f"{Counter(y_rv)}")
    evaluate_classifier(clf, X_rv, y_rv, threshold=threshold)

    # Evaluate the performance of the resampled validation subset
    final_score_title = f"Valid. balanced subset (thres: {100*threshold:.1f})"
    y_rv_pred, y_rv_pred_proba = predict(clf, X_rv, threshold)
    auc_rv = metrics.roc_auc_score(y_rv, y_rv_pred_proba)
    final_roc_filepath = ax_plot_roc_curve(y_rv, y_rv_pred_proba, auc_rv, final_score_title)
    final_conf_filepath = ax_plot_confusion_matrix(y_rv, y_rv_pred, ["Repaid", "Past due"], final_score_title)
    cm = metrics.confusion_matrix(y_rv, y_rv_pred)  # pour alimenter mlflow en post traitement
    return (
        main_image_filepath, final_roc_filepath, final_conf_filepath,
        auc_rs, auc_s, auc, auc_v, auc_rv, cm
    )
