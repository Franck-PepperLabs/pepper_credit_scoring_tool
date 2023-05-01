from typing import Dict, Callable, Union, Any, Optional
from collections import Counter

from IPython.display import display

import pandas as pd
import numpy as np

from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from imblearn.base import BaseSampler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, average_precision_score,
    precision_recall_curve, fbeta_score, roc_curve
)
import sklearn as skl
import lightgbm as lgbm

import matplotlib.pyplot as plt

# from home_credit.lightgbm_kernel_v2 import display_importances
# from home_credit.load import save_submission
from pepper.debug import kv, tx, tl, stl


def default_imputation(data: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing values in a DataFrame using the median of each column.
    Missing values are first replaced with NaN values before imputation. 
    Only the training data (TARGET > -1) is used to fit the imputer.
    
    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame containing the data to be imputed.
    
    Returns
    -------
    A pandas DataFrame with imputed values.
    """
    # Replace infinite values with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Separate training and test data
    data_train = data[data.TARGET > -1]
    
    # Fit the imputer on the training data only
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_median.fit(data_train)
    
    # Impute missing values in the entire dataset
    data_imputed = pd.DataFrame(
        imp_median.transform(data),
        columns=data.columns, index=data.index
    )
    return data_imputed


def fit_facade(clf, X_y_train, X_y_valid, loss_func):
    if isinstance(clf, lgbm.LGBMClassifier):
        """
        X_train: the training features
        y_train: the training target variable
        eval_set: a list of tuples containing validation set features and target variable [(X_valid, y_valid)]
        eval_metric: the evaluation metric to use during training
        verbose: controls the verbosity of the training output
        early_stopping_rounds: stops training if the evaluation metric doesn't improve after early_stopping_rounds rounds
        sample_weight: sample weights for each training sample
        """
        clf.fit(
            *X_y_train,
            eval_set=[X_y_train, X_y_valid],
            eval_metric=loss_func,
            verbose=200, early_stopping_rounds=200
        )
    elif isinstance(clf, ClassifierMixin):
        clf.fit(*X_y_train)
    else:
        raise ValueError(f"Invalid classifier type{type(clf)}")


def predict_facade(clf, X):
    if isinstance(clf, lgbm.LGBMClassifier):
        return clf.predict(X, num_iteration=clf.best_iteration_)
    else:
        return clf.predict(X)


def predict_proba_facade(clf, X):
    if isinstance(clf, lgbm.LGBMClassifier):
        return clf.predict_proba(X, num_iteration=clf.best_iteration_)[:, 1]
    else:
        return clf.predict_proba(X)[:, 1]


def get_feat_imp_facade(clf):  # clf.feature_importances_,
    if isinstance(clf, (
        lgbm.LGBMClassifier,
        skl.ensemble.RandomForestClassifier
    )):
        return clf.feature_importances_
    elif isinstance(clf, skl.linear_model.LogisticRegression):
        return clf.coef_[0]
    # TOCO : Pour le dummy, on peut dire que les features ont une importance égale
    else:
        print("Error: Unsupported classifier type")



def require_probas(metric: Callable) -> bool:
    """Checks if a metric requires probabilistic predictions.

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
        roc_auc_score, brier_score_loss,
        average_precision_score, precision_recall_curve
    ]


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
    """Evaluates predictions using the given evaluation metrics.

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
        If >0, progress messages are printed.

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
    else:
        if verbosity > 0:
            print(f"Only one class present in `{y_true_name}`. "
                   "The scores are not defined in that case.")


def plot_roc_curve(y_true, y_pred_proba, overall_auc):
    fpr, tpr,_ = roc_curve(y_true, y_pred_proba)
    plt.plot(
        fpr, tpr,
        label=f"Model AUC = {overall_auc:0.4f}",
        color="magenta"
    )
    plt.plot(
        [0, 1], [0, 1], linestyle=':',
        label=f"Baseline AUC = 0.5",
        color="yellow"
    )
    plt.fill_between(fpr, fpr, tpr, color='magenta', alpha=0.2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


# Eval a model with KFold or Stratified KFold
def kfold_train_and_eval_model(
    data: pd.DataFrame,
    clf: ClassifierMixin,
    imb_sampler: Union[BaseSampler, None] = SMOTETomek(),
    scaler: Union[TransformerMixin, None] = MinMaxScaler(),
    loss_func: Union[
        Dict[str, Callable[[np.ndarray, np.ndarray], float]],
        None] = {"AUC": roc_auc_score},
    eval_metrics: Union[
        Dict[str, Callable[[np.ndarray, np.ndarray], float]],
        None] = {
        # "AUC": roc_auc_score,
        "F2": lambda y_valid, y_pred: fbeta_score(y_valid, y_pred, beta=2)
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
        loss_func = {"AUC": roc_auc_score}
    
    if eval_metrics is None:
        eval_metrics = {}
    eval_metrics.update(loss_func)

    # Default imputation, if necessary
    # (should have been handled by feature enginering)
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
            "importance": get_feat_imp_facade(clf),   # clf.feature_importances_,
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
        eval_metrics, scores, "overall",
        y_train, oof_preds_proba, oof_preds_discr,
        "y_train", f"Full", verbosity
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
        res.update({"trained_clf": clf})

    return res
