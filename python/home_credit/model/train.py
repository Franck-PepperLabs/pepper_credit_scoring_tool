
from typing import Optional

from sklearn import metrics
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold

import pandas as pd
import numpy as np

from home_credit.model.preprocessing import train_preproc

from pepper.debug import kv, tx, tl, stl

from home_credit.model.facade import (
    fit_facade,
    predict_facade,
    predict_proba_facade,
    get_feature_importances_facade
)

from home_credit.model.eval import log_eval_predictions_v2
from home_credit.model.pipeline import create_model_with_preprocessing

class ModelConfig:
    """
    Configuration object for model training and evaluation.

    Attributes
    ----------
    loss_func : dict, optional
        Dictionary of loss functions for optimization during training.
        Keys are strings indicating the loss name, and values are callables
        that take two arrays (true and predicted labels) and return a float.
    eval_metrics : dict, optional
        Dictionary of evaluation metrics for model evaluation.
        Keys are strings indicating the metric name, and values are callables
        that take two arrays (true and predicted labels) and return a float.
    num_folds : int, optional
        Number of folds to be used for cross-validation during training.
    stratified : bool, optional
        Whether to use stratified k-fold cross-validation or not.
    """

    def __init__(
        self,
        loss_func: dict = None,
        eval_metrics: dict = None,
        num_folds: int = 5,
        stratified: bool = False,
    ):
        """
        Initialize a ModelConfig object with optional configuration parameters.

        Parameters
        ----------
        loss_func : dict, optional
            Dictionary of loss functions for optimization during training.
        eval_metrics : dict, optional
            Dictionary of evaluation metrics for model evaluation.
        num_folds : int, optional
            Number of folds to be used for cross-validation during training.
        stratified : bool, optional
            Whether to use stratified k-fold cross-validation or not.
        """
        self.loss_func = (
            loss_func if loss_func is not None
            else {"AUC": metrics.roc_auc_score}
        )
        self.eval_metrics = (
            eval_metrics if eval_metrics is not None
            else {
                "F2": lambda y_true,
                y_pred: metrics.fbeta_score(y_true, y_pred, beta=2)
            }
        )
        """ TODO Intégrer la logique
        if loss_func is None:
            loss_func = {"AUC": metrics.roc_auc_score}

        if eval_metrics is None:
            eval_metrics = {}

        eval_metrics.update(loss_func)
        """
        self.num_folds = num_folds
        self.stratified = stratified


class Data:
    def __init__(self, df: pd.DataFrame):
        """
        Represents a data container with raw and split data.

        Parameters
        ----------
        df : pd.DataFrame
            The original data.
        """
        self.df = df
        # TODO prepare ? me semble relever de l'amont, donc ne pas concerner cette partie
        self.X = self.df.drop(columns="TARGET").to_numpy()
        self.y = df["TARGET"].to_numpy()
        self.feature_names = list(self.df.columns)
        self.feature_names.remove("TARGET")
        self.X_train = self.X[self.y > -1]  # pourquoi .copy()
        self.y_train = self.y[self.y > -1]  # .copy()
        self.X_test = self.X[self.y == -1]  # .copy()

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_data_subset(self, idx):
        return self.X[idx], self.y[idx] 
        


class ModelTrainingResults:
    """
    Represents the results of a trained model, including predictions,
    performance scores, and feature importances.

    Attributes
    ----------
    resamples : dict
        Dictionary containing resampled training data features and target.
    preds : dict
        Dictionary containing probabilistic and discrete predictions
        on out-of-fold (oof) validation data.
    scores : dict
        Dictionary containing performance scores.
    feat_imps : pandas.DataFrame
        DataFrame containing feature importances.
    trained_clf : ClassifierMixin or None, optional
        Trained classifier object, if available.
    """

    def __init__(
        self,
        # resamples: dict,
        preds_proba: Optional[dict] = None,
        preds_discr: Optional[dict] = None,
        scores: Optional[dict] = None,
        feature_importances: Optional[pd.DataFrame] = None,
        pipeline: Optional[Pipeline] = None,
        config: Optional[ModelConfig] = None,
        data: Optional[Data] = None,
    ):
        """
        Initialize a ModelResults object with the specified results.

        Parameters
        ----------
        resamples : dict
            Dictionary containing resampled training data features and target.
        preds : dict
            Dictionary containing probabilistic and discrete predictions
            on out-of-fold (oof) validation data.
        scores : dict
            Dictionary containing performance scores.
        feat_imps : pandas.DataFrame
            DataFrame containing feature importances.
        trained_clf : ClassifierMixin or None, optional
            Trained classifier object, if available.
        """
        # self.resamples = resamples
        self.preds_proba = preds_proba
        self.preds_discr = preds_discr
        self.scores = scores
        self.feature_importances = feature_importances
        self.pipeline = pipeline
        self.config = config
        self.data = data

    @staticmethod
    def create_empty_results(n_samples: int, config: ModelConfig):
        """
        Create an empty ModelTrainingResults object with default values for predictions,
        feature importances, and scores.

        Parameters
        ----------
        n_samples : int
            The number of samples in the dataset.
        config : ModelConfig
            Configuration settings for model training and evaluation.

        Returns
        -------
        ModelTrainingResults
            An empty ModelTrainingResults object.
        """
        results = ModelTrainingResults()
        results.preds_proba = np.zeros(n_samples)
        results.preds_discr = np.zeros(n_samples)
        
        """results.preds_proba = {
            "train": {"proba": np.zeros(n_samples)},
            "valid": {"proba": np.zeros(n_samples)},
            "test": {"proba": np.zeros(config.X_test_shape[0])},
        }
        results.preds_discr = {
            "train": {"discr": np.zeros(n_samples)},
            "valid": {"discr": np.zeros(n_samples)},
            "test": {"discr": np.zeros(config.X_test_shape[0])},
        }"""
        results.feature_importances = []
        results.scores = {
            score_name: {"over_folds": []}
            for score_name in config.eval_metrics.keys()
        }
        return results

    def save_inputs(self, pipeline: Pipeline, config: ModelConfig, data: Data):
        """
        Save the pipeline, configuration, and data to the ModelTrainingResults object.

        Parameters
        ----------
        pipeline : Pipeline
            Trained machine learning pipeline.
        config : ModelConfig
            Configuration settings for model training and evaluation.
        data : Data
            Data container with raw and split data.
        """
        self.pipeline = pipeline
        self.config = config
        self.data = data


def log_train_start(
    data: Data,
    clf: ClassifierMixin,
    config: ModelConfig,
    verbosity: int
) -> None:
    """
    Log the start of the training and evaluation process, including dataset details,
    classifier information, loss function, evaluation metrics, and cross-validation setup.

    Parameters
    ----------
    data : Data
        Data container with raw and split data.
    clf : ClassifierMixin
        The machine learning classifier being trained and evaluated.
    config : ModelConfig
        Configuration settings for the model training and evaluation.
    verbosity : int
        Verbosity level for log messages (0 for silent, 1 for minimal, 2 for detailed).

    Returns
    -------
    None
        This function logs the training start information.
    """
    # Print the shape of the training and test data
    tl(verbosity, "Starting train and eval of:")
    kv(verbosity, "Labeled dataset of shape", f"{data.df.shape}")
    kv(verbosity, "Features set of shape", f"{data.X.shape}")
    tl(verbosity, "With:")
    kv(verbosity, "Classifier", ""), tx(verbosity, clf) # TODO : kv with display arg
    kv(verbosity, "Loss function", ""), tx(verbosity, config.loss_func)
    kv(verbosity, "Eval metrics", ""), tx(verbosity, config.eval_metrics)
    kv(verbosity,
        f"On {config.num_folds} {'stratified ' if config.stratified else ''}KFolds", ""
    )


def log_one_fold_train_eval(
    results: ModelTrainingResults,
    fold_id: int,
    valid_idx: np.ndarray,
    y_valid: np.ndarray,
    verbosity: int
) -> None:
    """
    Log the training and evaluation results for one fold during cross-validation.

    Parameters
    ----------
    results : ModelTrainingResults
        Container to store the results of model training and evaluation.
    fold_id : int
        The current fold's identifier.
    valid_idx : np.ndarray
        Validation data index array for the current fold.
    y_valid : np.ndarray
        True labels for the validation data.
    verbosity : int
        Verbosity level for log messages (0 for silent, 1 for minimal, 2 for detailed).

    Returns
    -------
    None
        This function logs the evaluation results for one fold.
    """
    log_eval_predictions_v2(
        config.eval_metrics,
        results.scores,
        "over_folds",
        y_valid,
        results.preds_proba[valid_idx],
        results.preds_discr[valid_idx],
        "y_valid",
        f"Fold {fold_id:2d}",
        verbosity
    )


def log_train_eval(
    config: ModelConfig,
    data: Data,
    results: ModelTrainingResults,
    verbosity: int
) -> None:
    """
    Log the overall training and evaluation results after cross-validation.

    Parameters
    ----------
    config : ModelConfig
        Configuration settings for the model training and evaluation.
    data : Data
        Data container with raw and split data.
    results : ModelTrainingResults
        Container to store the results of model training and evaluation.
    verbosity : int
        Verbosity level for log messages (0 for silent, 1 for minimal, 2 for detailed).

    Returns
    -------
    None
        This function logs the overall evaluation results after cross-validation.
    """
    log_eval_predictions_v2(
        config.eval_metrics,
        results.scores,
        "overall",
        data.y,
        results.preds_proba,
        results.preds_discr,
        "y_train",
        "Full",
        verbosity,
    )


# Cette fonction ne doit pas être là, mais dans le feature engineering amont
# ou encore dans l'impute de dernière minute incorporé dans le pipeline (mieux)
def prepare_data(data: pd.DataFrame, model: Pipeline):
    # Préparation des données (imputation, exclusion des colonnes non pertinentes, etc.)
    # Retourne X_train, y_train, X_test
    
    # Default config manager by ModelConfig
    
    # Preprocess the training data for machine learning.
    # Note : pas de scaling puisque celui-ci est intégré au pipeline du modèle
    X, y = train_preproc(data)  # C'est juste une imputation de sécurité
    # si le feature engineering a oublié de le faire
    
    # En fait, cela devient caduc, puisqu'embarqué dans le pipeline
    
    # TODO J'ai un doute sur le scaling : j'ai l'impression
    # qu'il est est fit sur train, mais qu'il n'est plus appliqué à test




def one_fold_train_and_eval(
    pipeline: Pipeline,
    config: ModelConfig,
    data: Data,
    results: ModelTrainingResults,
    split_indexes,
    fold_id, nfolds, verbosity,
):
    """
    Train and evaluate a machine learning model on one fold of cross-validation.

    Parameters
    ----------
    pipeline : Pipeline
        The machine learning pipeline to be trained and evaluated.
    config : ModelConfig
        Configuration settings for the model training and evaluation.
    data : Data
        Data container with raw and split data.
    results : ModelTrainingResults
        Container to store the results of model training and evaluation.
    split_indexes : tuple
        A tuple containing the training and validation index arrays.
    fold_id : int
        The current fold's identifier.
    nfolds : int
        The total number of folds in the cross-validation.
    verbosity : int
        Verbosity level for log messages (0 for silent, 1 for minimal, 2 for detailed).

    Returns:
    --------
    None:
        The function updates the 'results' object with predictions,
        feature importances, and scores.
    """
    stl(verbosity, f"Evaluate the {fold_id+1}-th fold (on {nfolds})")
    
    # Split the training and validation data using the current fold
    train_idx, valid_idx = split_indexes
    X_y_train = data.get_data_subset(train_idx)
    X_y_valid = data.get_data_subset(valid_idx)
    X_valid, y_valid = X_y_valid

    # Train the classifier using the training and validation data and the loss function
    fit_facade(pipeline, X_y_train, X_y_valid, config.loss_func)

    # Make predictions on the validation and test data using the trained model
    results.preds_proba[valid_idx] = predict_proba_facade(pipeline, X_valid)
    results.preds_discr[valid_idx] = predict_facade(pipeline, X_valid)

    # Get the feature importances for the current fold
    feature_importances = pd.DataFrame({
        "feature": data.feature_names,
        "importance": get_feature_importances_facade(pipeline),
        "fold": fold_id
    })

    # Concatenate the feature importances across all folds
    results.feature_importances.append(feature_importances)

    # Log the scores for the current fold
    log_one_fold_train_eval(results, fold_id, valid_idx, y_valid, verbosity)


def multi_folds_train_and_eval(
    pipeline: Pipeline,
    data: Data,
    config: ModelConfig,
    verbosity: int
):
    """
    Perform model training and evaluation on multiple folds of cross-validation.

    Parameters
    ----------
    pipeline : Pipeline
        The machine learning pipeline to be trained and evaluated.
    data : Data
        Data container with raw and split data.
    config : ModelConfig
        Configuration settings for the model training and evaluation.
    verbosity : int
        Verbosity level for log messages (0 for silent, 1 for minimal, 2 for detailed).

    Returns
    -------
    None
        The function updates the 'results' object with predictions,
        feature importances, and scores for each fold.
    """
    # Split the target variable and the features for training and test data
    X_y_train = data.get_train_data()
    
    # Create the cross-validation model
    fold_params = {
        "n_splits": config.num_folds,
        "shuffle": True, 
        "random_state": 42
    }
    folds = (StratifiedKFold if config.stratified else KFold)(**fold_params)

    # Iterate through the folds
    for fold_id, split_indexes in enumerate(folds.split(*X_y_train)):
        one_fold_train_and_eval(
            pipeline, config, results,
            data, split_indexes,
            fold_id, config.num_folds, verbosity,
        )


def train_and_eval_pipeline(
    raw_data: pd.DataFrame,
    pipeline: Pipeline,
    config: ModelConfig,
    verbosity: int = 0,
    add_inputs_in_results: bool = False,
) -> ModelTrainingResults:
    """
    Train and evaluate a machine learning pipeline using cross-validation.

    Parameters:
    -----------
    raw_data : pd.DataFrame
        The raw input data.
    pipeline : Pipeline
        The machine learning pipeline to be trained and evaluated.
    config : ModelConfig
        Configuration settings for the model training and evaluation.
    verbosity : int, optional
        Verbosity level for log messages (0 for silent, 1 for minimal, 2 for detailed).
    add_inputs_in_results : bool, optional
        Whether to add the input pipeline, config, and data to the results object.

    Returns:
    --------
    ModelTrainingResults
        The results of the model training and evaluation.
    """
    
    # Initialize the data container
    data = Data(raw_data)
    
    # Log the start of the training
    log_train_start(pipeline, config, data, verbosity)
    
    # Create the structure to store the results
    results = ModelTrainingResults.create_empty_results(data.X.shape[0])
    
    # Train and evaluate the pipeline using cross-validation
    multi_folds_train_and_eval(pipeline, data, config, verbosity)

    # Concatenate the feature importances across all folds
    results.feature_importances = pd.concat(results.feature_importances, axis=0)

    # Log the final evaluation results
    log_train_eval(config, data, results, verbosity)

    # Optionally, add inputs to the results object
    if add_inputs_in_results:
        results.save_inputs(pipeline, config, data)

    return results


if __name__ == "__main__":
    verbosity = 1
    add_inputs_in_results = True
    data = None # Input as output of feature engineering
    pipeline = create_model_with_preprocessing(
        ...
    )
    config = ModelConfig(
        loss_func={"AUC": metrics.roc_auc_score},
        eval_metrics={
            "F2": lambda y_true,
            y_pred: metrics.fbeta_score(y_true, y_pred, beta=2)
        },
        num_folds=5,
        stratified=False
    )
    results = train_and_eval_pipeline(
        data, pipeline, config,
        add_inputs_in_results, verbosity
    )
