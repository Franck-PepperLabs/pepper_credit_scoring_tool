from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from imblearn.base import BaseSampler


def create_model_with_preprocessing(
    clf: ClassifierMixin,
    scaler: TransformerMixin = None,
    resampler: BaseSampler = None
) -> Pipeline:
    """
    Create a machine learning model with optional preprocessing steps.

    Parameters
    ----------
    clf : ClassifierMixin
        The classifier to be used as the main model.
    scaler : TransformerMixin or None, optional
        The scaler to be used for feature scaling.
        If None, no scaling is applied.
    resampler : BaseSampler or None, optional
        The resampler to be used for class imbalance correction.
        If None, no resampling is applied.

    Returns
    -------
    model : Pipeline
        A scikit-learn pipeline that includes the specified preprocessing steps
        (scaler and resampler) followed by the classifier.
    """

    # Define the steps of the pipeline
    steps = []

    # Add the scaler step if provided
    if scaler is not None:
        steps.append(("scaler", scaler))

    # Add the resampler step if provided
    if resampler is not None:
        steps.append(("resampler", resampler))

    # Add the classifier as the final step
    steps.append(("classifier", clf))

    return Pipeline(steps)
