from typing import Optional, Dict, Callable

import inspect, hashlib, json, re, os
import pandas as pd
import numpy as np

from pepper.env import get_tmp_dir
from pepper.utils import create_if_not_exist
from pepper.np_utils import reconstruct_ndarray  #, ndarray_to_list
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
    # Create folder names based on class_name and method_name
    class_folder = camel_to_snake(class_name)
    method_folder = method_name
    
    # Attempt to generate the hash code for the arguments
    hashed_code = hash_code(arguments)
    
    # Ensure class_folder and method_folder exist in the index
    if class_folder not in _persist_index:
        _persist_index[class_folder] = {}
    class_index = _persist_index[class_folder]
    if method_folder not in class_index:
        class_index[method_folder] = {}

    # Add the entry to the method_index using the hash code as the key
    method_index = class_index[method_folder]
    method_index[hashed_code] = arguments
    
    # Save the updated persistence index
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


""" Load and save
"""


def has_more_than_one_dim(x: np.ndarray) -> bool:
    """
    Check if an ndarray has more than one dimension.

    Parameters:
    -----------
    x : np.ndarray
        The numpy array to check.

    Returns:
    --------
    bool
        True if the array has more than one dimension, False otherwise.

    Example:
    --------
    >>> arr = np.array([1, 2, 3])
    >>> has_more_than_one_dim(arr)
    False
    >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
    >>> has_more_than_one_dim(arr)
    True
    """
    return isinstance(x, np.ndarray) and x.ndim > 1


def is_multi_dim_array_col(
    s: pd.Series,
    weak_check: bool = False
) -> bool:
    """
    Check if a Pandas Series contains multi-dimensional numpy arrays in its elements.

    Parameters:
    -----------
    s : pd.Series
        The Pandas Series to check.

    weak_check : bool, optional (default=False)
        If True, performs a weak check by examining only the first element of the Series.
        If False, checks all elements in the Series.

    Returns:
    --------
    bool
        True if the Series contains multi-dimensional arrays, False otherwise.

    Example:
    --------
    >>> s = pd.Series([np.array([1, 2, 3]), np.array([[4, 5], [6, 7]])])
    >>> is_multi_dim_array_col(s)
    True
    >>> s = pd.Series([np.array([1, 2, 3]), np.array([4, 5, 6])])
    >>> is_multi_dim_array_col(s)
    False
    >>> s = pd.Series([np.array([1, 2, 3])])
    >>> is_multi_dim_array_col(s, weak_check=True)
    False
    """
    if s.empty:
        return False
    if weak_check:
        return has_more_than_one_dim(s.iloc[0])
    return s.apply(has_more_than_one_dim).any()


def encode_ndarrays_for_parquet(
    data: pd.DataFrame,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Encode multi-dimensional numpy arrays in a DataFrame for Parquet serialization.

    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame containing columns to be encoded.

    inplace : bool, optional (default=False)
        If True, the DataFrame will be modified in-place.
        If False, a copy of the DataFrame will be created.

    Returns:
    --------
    pd.DataFrame
        The modified DataFrame with encoded columns.

    Example:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame({'A': [np.array([1, 2, 3]), np.array([[4, 5], [6, 7]])],
    ...                      'B': [np.array([8, 9]), np.array([[10], [11]])]})
    >>> encoded_data = encode_ndarrays_for_parquet(data)
    >>> encoded_data.columns
    Index(['A_flat', 'A_shape', 'B_flat', 'B_shape'], dtype='object')

    This function encodes multi-dimensional numpy arrays in the DataFrame
    by replacing each ndarray column 'A'with two columns 'A_flat' and 'A_shape'
    representing the flattened array and its original shape, respectively.
    """

    if not inplace:
        data = data.copy()
    # Iterate through columns and identify those containing multi-dimensional ndarrays
    # For each such column 'A', replace it with the column pair 'A_flat' and 'A_shape'
    for col in data.columns:
        s = data[col]
        # Note: We can't use 'for loc, col in enumerate(data.columns)'
        # since modifying 'data.columns' internally in the loop
        loc = data.columns.get_loc(col)
        if is_multi_dim_array_col(s):
            data.insert(loc, f"{col}_flat", s.apply(np.ravel))
            data.insert(loc, f"{col}_shape", s.apply(np.shape))
            data.drop(columns=col, inplace=True)
    return data


def decode_ndarrays_from_parquet(
    data: pd.DataFrame,
    inplace: bool = True
) -> pd.DataFrame:
    """
    Decode ndarrays that were encoded for Parquet storage.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing columns with encoded ndarrays.
    inplace : bool, optional
        If True, decode ndarrays in-place (default is True).

    Returns
    -------
    pd.DataFrame
        The DataFrame with ndarrays decoded.

    Notes
    -----
    This function decodes ndarrays that were previously encoded for storage
    in Parquet files. It identifies pairs of columns in the DataFrame,
    where one column represents the shape of an ndarray, and the other column
    represents the flattened ndarray data. It then decodes these columns and
    replaces them with the original ndarray columns.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'A_shape': [(2, 3), (3, 2)],
    ...     'A_flat': [
    ...         np.array([1, 2, 3, 4, 5, 6]),
    ...         np.array([7, 8, 9, 10, 11, 12])
    ...     ]
    ... })
    >>> decoded_data = decode_ndarrays_from_parquet(data)
        A
    0	[[1, 2, 3], [4, 5, 6]]
    1	[[7, 8], [9, 10], [11, 12]]
    """
    if not inplace:
        data = data.copy()

    # Start by identifying all the pairs of columns A_shape, A_flat
    # 1/ Form the `shape_cols` list of A for each A_shape column 
    # 2/ Form the `flat_cols` list of A for each A_flat column
    # 3/ Produce `new_cols` as the intersection of the two lists
    shape_cols = [col.replace("_shape", "") for col in data.columns if col.endswith("_shape")]
    flat_cols = [col.replace("_flat", "") for col in data.columns if col.endswith("_flat")]
    new_cols = list(set(shape_cols).intersection(flat_cols))

    # For each of them, replace a column A with A_flat.reshape(A_shape)
    def decode_ndarray(row):
        shape = row[0]
        flat = row[1]
        return flat.reshape(shape)
    
    # For each column pair that needs decoding
    for new_col in new_cols:
        # Get the location (index) of the shape column
        loc = data.columns.get_loc(f"{new_col}_shape")

        # Identify the old shape and flat columns
        old_cols = [f"{new_col}_shape", f"{new_col}_flat"]

        # Decode the ndarray by applying the decode_ndarray function to each row
        decoded = data[old_cols].apply(decode_ndarray, axis=1)

        # Insert the decoded ndarray column at the original location
        data.insert(loc, new_col, decoded)

        # Drop the old shape and flat columns as they are no longer needed
        data.drop(columns=old_cols, inplace=True)

    return data


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
    
    Note
    ----
    This function converts numpy arrays in the DataFrame to lists of lists
    to avoid 'ArrowInvalid' issues when saving to Parquet. This conversion
    ensures that the data can be correctly stored and loaded.
    """
    # Define the folder path based on naming conventions
    folder_path = os.path.join(
        get_persist_dir(),
        camel_to_snake(class_name),
        method_name
    )
    # Create the folder if it doesn't exist
    create_if_not_exist(folder_path)
    
    # Define the file path with a unique hash code
    file_path = os.path.join(folder_path, f"{hash_code(arguments)}.pqt")
    
    # Convert numpy arrays to lists of lists to avoid 'ArrowInvalid' issue
    # data = data.applymap(lambda x: list(x) if isinstance(x, np.ndarray) else x)
    encoded_data = encode_ndarrays_for_parquet(data)
    
    # Save the DataFrame to Parquet
    print(f"Save to {file_path}")
    encoded_data.to_parquet(file_path, engine="pyarrow", compression="gzip")


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
        
    Note
    ----
    This function converts lists of lists in the loaded DataFrame back to numpy
    arrays to ensure consistency with the original data structure. This conversion
    is necessary due to the use of applymap during the saving process.
    """
    # Construct the folder path based on class and method names
    folder_path = os.path.join(
        get_persist_dir(),
        camel_to_snake(class_name),
        method_name
    )
    
    # Generate the full file path
    file_path = os.path.join(folder_path, f"{hash_code(arguments)}.pqt")

    # Check if the Parquet file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    # Load the data from the Parquet file
    loaded_data = pd.read_parquet(file_path, engine="pyarrow")

    # Convert lists of lists back to numpy arrays
    # loaded_data = loaded_data.applymap(
    #     lambda x: reconstruct_ndarray(x) if isinstance(x, np.ndarray) else x
    # )
    decode_ndarrays_from_parquet(loaded_data)
    
    return loaded_data


def controlled_load(
    method: str,
    kwargs: Dict,
    builder: Callable,
    in_cache_name: Optional[str] = None,
    debug: Optional[bool] = False
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
    from_file = from_file and not update_file

    # Get the parameters for persistence and debugging information
    params = get_persist_params(kwargs, method, debug)

    # Check if loading from a file is requested and the data exists in the index
    if from_file and is_in_index(*params):
        # Define a loader function for loading from file
        loader = lambda: load_from_parquet(*params)
        # Return loaded data or initialize it in the cache
        return loader() if no_cache else Cache.init(in_cache_name, loader)
    
    # If not loading from file or data is not in the index, build the data
    data = builder() if no_cache else Cache.init(in_cache_name, builder)

    # If loading from file or updating the file is requested, save to file and update the index
    if from_file or update_file:
        save_to_parquet(data, *params)
        add_entry_to_index(*params)

    return data

