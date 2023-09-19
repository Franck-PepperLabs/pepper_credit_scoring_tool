from typing import Optional, Dict, Callable

import inspect, hashlib, json, re, os
import pandas as pd

from pepper.env import get_tmp_dir
from pepper.utils import create_if_not_exist
from pepper.cache import Cache


def this_f_name() -> str:
    """
    Return the name of the calling function.

    Returns
    -------
    str: The name of the calling function as a string.

    Example
    -------
    >>> def my_function():
    ...     return this_f_name()
    >>> my_function()
    'my_function'
    """
    return inspect.stack()[1][3]


def pop_container_class(kwargs: dict) -> str:
    """
    Remove and return the container class name from keyword arguments.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments passed to a function.

    Returns
    -------
    str
        The name of the container class, if found; otherwise, None.
    """
    class_name = None
    first_arg = list(kwargs.keys())[0]
    if first_arg == "cls":
        class_name = kwargs[first_arg].__name__
    elif first_arg == "self":
        class_name = type(kwargs[first_arg]).__name__
    if class_name:
        kwargs.pop(first_arg)
    return class_name


def hash_code(kwargs: dict) -> str:
    """
    Generate a unique hash code for a dictionary of keyword arguments.

    Parameters
    ----------
    kwargs : dict
        The dictionary of keyword arguments.

    Returns
    -------
    str
        The unique hash code as a hexadecimal string.

    Example
    -------
    >>> hash_code({"a": 1, "b": "hello"})
    'b37e0f96e7b5b9e54d69804657c53a9583e0a98cb1a87228c02f2f237f80e1ed'
    """
    params_str = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha256(params_str.encode()).hexdigest()


def camel_to_snake(name: str) -> str:
    """
    Convert a camelCase string to snake_case.

    Parameters
    ----------
    name : str
        The camelCase string to convert.

    Returns
    -------
    str
        The snake_case string.

    Example
    -------
    >>> camel_to_snake("myCamelString")
    'my_camel_string'
    """
    if not name:
        return None
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def f_info(
    class_name: str,
    f_name: str,
    kwargs: dict,
    subdir: str,
    hcode: str
) -> None:
    """
    Print information about a function and its arguments.

    Parameters
    ----------
    class_name : str
        The name of the class containing the function.
    f_name : str
        The name of the function.
    kwargs : dict
        The dictionary of function arguments.
    subdir : str
        The subdirectory where data will be persisted.
    hcode : str
        The hash code for the function arguments.

    Returns
    -------
    None
    """
    print(f"container class name: {class_name}")
    print(f"f_name: {f_name}")
    print(f"kwargs: {kwargs}")
    print(f"hcode: {hcode}")
    print(f"subdir: {subdir}")


def get_persist_params(
    kwargs: dict,
    f_name: str,
    debug = False
) -> tuple:
    """
    Get parameters for persisting function results.

    Parameters
    ----------
    kwargs : dict
        Dictionary of function arguments.
    f_name : str
        The name of the function.

    Returns
    -------
    tuple
        A tuple containing class name, function name, and function arguments.
    """
    class_name = pop_container_class(kwargs)
    kwargs.pop("no_cache", None)
    kwargs.pop("from_file", None)
    kwargs.pop("update_file", None)
    if debug:
        f_info(
            class_name, f_name, kwargs,
            camel_to_snake(class_name),
            hash_code(kwargs)
        )
    return class_name, f_name, kwargs


def get_persist_dir() -> str:
    """
    Return the project's `persist` directory path.

    Returns
    -------
    str
        The project's modules directory path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.
    """
    return os.path.join(get_tmp_dir(), "persist")


def load_persist_index() -> dict:
    """
    Load the persistence index from the `persist` directory.

    Returns
    -------
    dict
        The persistence index as a dictionary,
        or an empty dictionary if the index file does not exist.
    """
    index_path = os.path.join(get_persist_dir(), "index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as index_file:
            return json.load(index_file)
    return {}


_persist_index = load_persist_index()


def save_persist_index(index_data: dict):
    """
    Save the persistence index to the `persist` directory.

    Parameters
    ----------
    index_data : dict
        The persistence index to be saved as a dictionary.
    """
    index_path = os.path.join(get_persist_dir(), "index.json")
    with open(index_path, "w") as index_file:
        json.dump(index_data, index_file, indent=4)


def add_entry_to_index(
    class_name: str,
    method_name: str,
    arguments: dict
) -> None:
    """
    Add an entry to the persistence index.

    Parameters
    ----------
    class_name : str
        The name of the class associated with the method.
    method_name : str
        The name of the method or function.
    arguments : dict
        The arguments used for the method or function.

    Returns
    -------
    None
    """
    class_folder = camel_to_snake(class_name)
    method_folder = method_name
    if class_folder not in _persist_index:
        _persist_index[class_folder] = {}
    class_index = _persist_index[class_folder]
    if method_folder not in class_index:
        class_index[method_folder] = {}
    method_index = class_index[method_folder]
    method_index[hash_code(arguments)] = arguments
    save_persist_index(_persist_index)


def is_in_index(
    class_name: str,
    method_name: str,
    arguments: dict
) -> bool:
    """
    Check if an entry exists in the persistence index.

    Parameters
    ----------
    class_name : str
        The name of the class associated with the method.
    method_name : str
        The name of the method or function.
    arguments : dict
        The arguments used for the method or function.

    Returns
    -------
    bool
        True if the entry exists in the index, False otherwise.
    """
    class_folder = camel_to_snake(class_name)
    method_folder = method_name
    if class_folder in _persist_index:
        class_index = _persist_index[class_folder]
        if method_folder in class_index:
            method_index = class_index[method_folder]
            return hash_code(arguments) in method_index
    return False


def save_to_parquet(
    data: pd.DataFrame,
    class_name: str,
    method_name: str,
    arguments: dict
) -> None:
    """
    Save a DataFrame to a Parquet file with a specific naming convention.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be saved.
    class_name : str
        The name of the class associated with the data.
    method_name : str
        The name of the method associated with the data.
    arguments : dict
        A dictionary containing the arguments used to generate the data.

    Returns
    -------
    None
    """
    folder_path = os.path.join(
        get_persist_dir(),
        camel_to_snake(class_name),
        method_name
    )
    create_if_not_exist(folder_path)
    file_path = os.path.join(folder_path, f"{hash_code(arguments)}.pqt")
    print(f"Save to {file_path}")
    data.to_parquet(file_path, engine="pyarrow", compression="gzip")


def load_from_parquet(
    class_name: str,
    method_name: str,
    arguments: dict
) -> pd.DataFrame:
    """
    Load a DataFrame from a Parquet file using specific naming convention.

    Parameters
    ----------
    class_name : str
        The name of the class associated with the data.
    method_name : str
        The name of the method associated with the data.
    arguments : dict
        A dictionary containing the arguments used to generate the data.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the Parquet file does not exist.
    """
    folder_path = os.path.join(
        get_persist_dir(),
        camel_to_snake(class_name),
        method_name
    )
    file_path = os.path.join(folder_path, f"{hash_code(arguments)}.pqt")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    return pd.read_parquet(file_path, engine="pyarrow")


def controlled_load(
    method: str,
    kwargs: Dict,
    builder: Callable,
    in_cache_name: Optional[str] = None,
    debug: Optional[bool] = True
) -> pd.DataFrame:
    """
    Control the loading of data based on cache and file operations.

    Parameters
    ----------
    method : str
        The name of the method or function requesting data loading.
    kwargs : Dict
        A dictionary of function arguments.
    builder : Callable
        A function that constructs and returns the data when not found in cache or file.
    in_cache_name : str, optional
        The name under which to store data in the cache (default: None).
    debug : bool, optional
        If True, print debugging information (default: True).

    Returns
    -------
    pd.DataFrame
        The loaded or constructed DataFrame.

    Notes
    -----
    This function provides control over the loading of data
    by considering cache, file, and construction options.
    It checks for cache, file, and construction needs based
    on the provided parameters and returns the appropriate data.

    """
    # Extract cache-related parameters
    no_cache = kwargs.pop("no_cache", None)
    from_file = kwargs.pop("from_file", None)
    update_file = kwargs.pop("update_file", None)

    # Get the parameters for persistence and debugging information
    params = get_persist_params(kwargs, method, debug)

    # Check if loading from a file is requested and the data exists in the index
    if from_file and is_in_index(*params):
        # Define a loader function for loading from file
        loader = lambda: load_from_parquet(*params)
        # Return loaded data or initialize it in the cache
        return loader() if no_cache else Cache.init(in_cache_name, loader)

    # TODO après les cas simples,
    # tester sur les cas où il faut passer au builder
    # des arguments supplémentaires
    
    # If not loading from file or data is not in the index, build the data
    data = builder() if no_cache else Cache.init(in_cache_name, builder)

    # If loading from file or updating the file is requested, save to file and update the index
    if from_file or update_file:
        save_to_parquet(data, *params)
        add_entry_to_index(*params)

    return data

