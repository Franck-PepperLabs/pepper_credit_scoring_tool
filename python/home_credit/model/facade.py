"""
Module: home_credit/model/facade.py

This module provides a collection of utility functions and facades
for common tasks related to machine learning model handling and interactions.
It offers a unified interface for handling various classifiers,
imputing missing values, making predictions, and retrieving feature importances.

Functions:
- `fit_facade(clf: Union[lgbm.LGBMClassifier, ClassifierMixin], X_y_train: Tuple[pd.DataFrame, pd.Series], X_y_valid: Tuple[pd.DataFrame, pd.Series], loss_func: Callable) -> None`:
    Fit a classifier using the appropriate training method based on the classifier type.

- `predict_facade(clf: Union[lgbm.LGBMClassifier, ClassifierMixin], X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray`:
    Facade function for making predictions using a classifier.

- `predict_proba_facade(clf: Union[lgbm.LGBMClassifier, Any], X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray`:
    Predict class probabilities using a classifier.

- `get_feature_importances_facade(clf: Union[lgbm.LGBMClassifier, RandomForestClassifier, LogisticRegression, DummyClassifier]) -> List[float]`:
    Retrieve feature importances from various classifiers.

Supported Classifiers:
- LightGBM Classifier
- Random Forest Classifier
- Logistic Regression
- Dummy Classifier (for feature importance equalization)

This module aims to simplify and standardize machine learning workflows
when working with different classifier types, making it easier to handle
common tasks during model development.

Example:
>>> from lightgbm import LGBMClassifier
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.metrics import log_loss
>>> X, y = load_iris(return_X_y=True)
>>> X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
>>> clf = LGBMClassifier()
>>> fit_facade(clf, (X_train, y_train), (X_valid, y_valid), log_loss)

For more details on each function and its usage, please refer
to the function-specific docstrings and examples.
"""
from typing import Tuple, List, Any, Union, Callable

import pandas as pd
import numpy as np

from sklearn.base import ClassifierMixin

import sklearn as skl
import lightgbm as lgbm

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def fit_facade(
    clf: Union[lgbm.LGBMClassifier, ClassifierMixin],
    X_y_train: Tuple[pd.DataFrame, pd.Series],
    X_y_valid: Tuple[pd.DataFrame, pd.Series],
    loss_func: Callable
) -> None:
    """
    Fit a classifier using the appropriate training method
    based on the classifier type.

    Parameters:
    -----------
    clf : Union[lgbm.LGBMClassifier, ClassifierMixin]
        The classifier to fit. Should be either a LightGBM classifier
        or a scikit-learn compatible classifier.

    X_y_train : Tuple[pd.DataFrame, pd.Series]
        A tuple containing the training features
        and the corresponding target variable.

    X_y_valid : Tuple[pd.DataFrame, pd.Series]
        A tuple containing the validation features
        and the corresponding target variable.

    loss_func : Callable
        The loss function used for evaluation during training.

    Raises:
    -------
    ValueError
        If an invalid classifier type is provided.

    Notes:
    ------
    - If the classifier is an LGBMClassifier, it will use early stopping during
        training with evaluation sets provided by `X_y_train` and `X_y_valid`.
    - If the classifier is a scikit-learn compatible classifier,
        it will fit the model without early stopping.

    Example:
    --------
    >>> from lightgbm import LGBMClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import log_loss
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    >>> clf = LGBMClassifier()
    >>> fit_facade(clf, (X_train, y_train), (X_valid, y_valid), log_loss)
    """
    if isinstance(clf, lgbm.LGBMClassifier):
        clf.fit(
            *X_y_train,
            eval_set=[X_y_train, X_y_valid],
            eval_metric=loss_func,
            verbose=200, early_stopping_rounds=200
        )
    elif isinstance(clf, ClassifierMixin):
        clf.fit(*X_y_train)
    else:
        raise ValueError(f"Invalid classifier type: {type(clf)}")


def predict_facade(
    clf: Union[lgbm.LGBMClassifier, ClassifierMixin],
    X: Union[np.ndarray, pd.DataFrame]
) -> np.ndarray:
    """
    Facade function for making predictions using a classifier.

    Parameters
    ----------
    clf : Union[lgbm.LGBMClassifier, ClassifierMixin]
        The classifier object to use for making predictions.

    X : Union[np.ndarray, pd.DataFrame], shape (n_samples, n_features)
        The input features for which predictions should be made.

    Returns
    -------
    np.ndarray
        The predicted class labels or values.

    Notes
    -----
    This function serves as a unified interface for making predictions
    using different classifiers.
    If the classifier is of type `LGBMClassifier`, it uses the `predict`
    method with `num_iteration` set to the best iteration.
    For other classifiers, it uses the standard `predict` method.

    Examples
    --------
    >>> from lightgbm import LGBMClassifier
    >>> clf = LGBMClassifier()
    >>> X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    >>> predict_facade(clf, X)
    array([0, 1])
    """
    if isinstance(clf, lgbm.LGBMClassifier):
        return clf.predict(X, num_iteration=clf.best_iteration_)
    else:
        return clf.predict(X)


def predict_proba_facade(
    clf: Union[lgbm.LGBMClassifier, Any],
    X: Union[np.ndarray, pd.DataFrame]
) -> np.ndarray:
    """
    Predict class probabilities using a classifier.

    This function takes a classifier and input data,
    and predicts class probabilities for binary classification tasks.

    Parameters:
    -----------
    clf : Union[lgbm.LGBMClassifier, Any]
        The classifier model. Should be an instance of either LGBMClassifier
        or a compatible classifier.
    
    X : Union[np.ndarray, pd.DataFrame]
        The input data on which to make predictions.
        It can be either a NumPy array or a Pandas DataFrame.

    Returns:
    --------
    np.ndarray
        An array of predicted class probabilities.
        For binary classification, this will be an array of shape (n_samples,),
        where n_samples is the number of samples in the input data.
        Each value represents the probability of the positive class.

    Example:
    --------
    >>> from lightgbm import LGBMClassifier
    >>> import numpy as np
    >>> clf = LGBMClassifier()
    >>> X_test = np.array([[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]])
    >>> predictions = predict_proba_facade(clf, X_test)
    >>> print(predictions)
    [0.97546522 0.03874425]
    """
    if isinstance(clf, lgbm.LGBMClassifier):
        return clf.predict_proba(X, num_iteration=clf.best_iteration_)[:, 1]
    else:
        return clf.predict_proba(X)[:, 1]


def get_feature_importances_facade(
    clf: Union[
        lgbm.LGBMClassifier,
        RandomForestClassifier,
        LogisticRegression,
        DummyClassifier
    ]
) -> List[float]:
    """
    Get feature importances from a classifier.

    This function retrieves feature importances from various classifiers,
    including LightGBM, RandomForest, Logistic Regression,
    and DummyClassifier.

    Parameters:
    -----------
    clf : object
        The classifier from which to retrieve feature importances.

    Returns:
    --------
    feature_importances : list
        A list containing feature importances.
        The format may vary depending on the classifier type.
        For LGBMClassifier and RandomForestClassifier,
        it returns feature importance scores.
        For LogisticRegression, it returns the coefficients.
        For DummyClassifier, it returns equal importances for all features.

    Raises:
    -------
    ValueError:
        If the classifier type is not supported.

    Example:
    --------
    >>> clf = LGBMClassifier()
    >>> importances = get_feature_importances_facade(clf)
    >>> print(importances)
    [0.1, 0.2, 0.3, 0.4]
    """

    if isinstance(clf, (lgbm.LGBMClassifier, RandomForestClassifier)):
        return clf.feature_importances_
    elif isinstance(clf, LogisticRegression):
        return clf.coef_[0]
    elif isinstance(clf, DummyClassifier):
        raise ValueError(
            "DummyClassifier does not support feature importances. "
            "Use `[1.0 / n_features] * n_features` instead."
        )
    else:
        raise ValueError(f"Unsupported classifier type: {type(clf)}")
