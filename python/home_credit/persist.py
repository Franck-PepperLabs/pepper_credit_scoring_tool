import os
import io
import warnings
from typing import Union, Tuple, Type, Optional
import pickle, joblib, skops.io
from pepper.env import get_tmp_dir
from pepper.utils import create_if_not_exist
from sklearn.base import ClassifierMixin
from lightgbm import LGBMClassifier


def _validate_save_engine_and_get_ext(
    save_engine: object = joblib,   # Union[pickle, joblib, skops.io]
) -> str:
    """Validates the save engine and return its file extension.

    Parameters
    ----------
    save_engine : object, optional
        The save engine to validate (`pickle`, `joblib`, or `skops.io`).
        Default: joblib.

    Raises
    ------
    ValueError
        If the save engine is not supported.

    Returns
    -------
    str
        The file extension for the given save engine.
    """
    ext_dict = {pickle: "pkl", joblib: "jbl", skops.io: "sio"}
    ext = ext_dict.get(save_engine)
    if ext is None:
        raise ValueError(
            f"Invalid save engine specified: {save_engine}. "
            f"Supported engines are: {list(ext_dict.keys())}"
        )
    return ext


def _get_trusted_model_types() -> Tuple[Type]:
    """Returns a tuple of trusted machine learning model types.

    Returns
    -------
    Tuple[Type]
        A tuple containing the trusted machine learning model types.
    """
    return (ClassifierMixin, LGBMClassifier)


def _validate_model_type(model_type: Type) -> None:
    """Validates the type of the machine learning model.

    Parameters
    ----------
    model_type : Type
        The machine learning model type to validate.

    Raises
    ------
    TypeError
        If the model type is not a trusted type.

    Returns
    -------
    None
    """
    trusted_model_types = _get_trusted_model_types()
    if not issubclass(model_type, trusted_model_types):
        raise TypeError(
            f"Invalid model type: {model_type}. "
            f"Must be an instance of either {trusted_model_types}."
        )


def _get_model_filepath(save_dir, model_name, ext):
    """Returns the file path for the model given the directory, name, and extension.

    Parameters
    ----------
    save_dir : str or None
        The directory where the model is to be saved.
        If None, the temporary directory is used.
    model_name : str
        The name of the model.
    ext : str
        The file extension for the model.

    Returns
    -------
    str
        The file path for the model.
    """
    save_dir = (
        os.path.join(get_tmp_dir(), "persist")
        if save_dir is None else save_dir
    )
    model_dir = os.path.join(save_dir, model_name)
    create_if_not_exist(model_dir)
    return os.path.join(model_dir, f"model.{ext}")


def save_model(
    model: object,
    model_name: str,
    save_engine: Optional[object] = joblib,   # Union[pickle, joblib, skops.io]
    save_dir: Optional[str] = None
) -> None:
    """Saves a machine learning model to disk using either `pickle`, `joblib`, or `skops.io`.

    Parameters
    ----------
    model : object
        A machine learning model object.
    model_name : str
        The name of the model.
    save_engine : {pickle, joblib, skops.io}, optional
        The save engine to use. Default is `joblib`.
    save_dir : str, optional
        The directory to save the model in.
        If None, the temporary directory is used. Default is None.

    Raises
    ------
    ValueError
        If the save engine is not supported.
    TypeError
        If the model is not a trusted type.

    Returns
    -------
        None.
    """
    # Validate the save engine and get the extension
    ext = _validate_save_engine_and_get_ext(save_engine)

    # Validate the model type
    _validate_model_type(type(model))
         
    # Determine the save location and file path
    model_filepath = _get_model_filepath(save_dir, model_name, ext)

    # Save
    with open(model_filepath, "wb") as f:
        save_engine.dump(model, f)


def load_model(
    model_name: str,
    save_engine: object = joblib,   # Union[pickle, joblib, skops.io]
    save_dir: str = None
) -> object:
    """Loads a machine learning model from disk using either `pickle`, `joblib`, or `skops.io`.

    Parameters
    ----------
    model_name : str
        The name of the model.
    save_engine : object, optional
        The save engine to use (`pickle`, `joblib`, or `skops.io`).
        Default: joblib.
    save_dir : str, optional
        The directory where the model is saved.
        If None, the temporary directory is used. Default: None.

    Raises
    ------
    ValueError
        If the save engine is not supported.

    Returns
    -------
    object
        The loaded machine learning model object.
    """
    # Validate the save engine
    ext = _validate_save_engine_and_get_ext(save_engine)

    # Determine the load location and file path
    model_filepath = _get_model_filepath(save_dir, model_name, ext)

    # Load
    #with open(model_filepath, "rb") as f:
    if save_engine is skops.io:
        model_type = save_engine.get_untrusted_types(file=model_filepath)
        # Validate the model type
        _validate_model_type(model_type)
        return save_engine.load(file=model_filepath, trusted=model_type)
    else:
        warnings.warn(
            "Using a non-skops.io save engine can potentially load malicious objects. "
            "It is recommended to use skops.io instead.",
            UserWarning
        )
        return save_engine.load(model_filepath)

