from typing import Optional, List, Any
import os

import pandas as pd
import numpy as np

from pepper.utils import bold, green, red
from pepper.env import get_project_dir, get_dataset_dir


def print_path_info(path: str) -> None:
    """
    Print information about a file or directory path.

    This function prints whether a given file or directory path exists within the
    project directory and provides additional formatting to highlight the project
    directory path, obfuscating sensitive user-specific information.

    Parameters:
    -----------
    path : str
        The file or directory path to check.

    Returns:
    --------
    None

    Notes:
    ------
    The function replaces the absolute path to the project directory with
    '[project_dir]' to protect sensitive user-specific information that may be
    present in the path.

    Examples:
    ---------
    >>> print_path_info('/path/to/some_file.csv')
    [project_dir]/path/to/some_file.csv exists

    >>> print_path_info('/nonexistent/path')
    [project_dir]/nonexistent/path doesn't exist
    """
    project_dir = get_project_dir()
    print(
        path.replace(project_dir, '[project_dir]'),
        'exists' if os.path.exists(path) else 'doesn\'t exist'
    )


def create_subdir(
    project_path: str,
    rel_path: str = "",
    verbose: bool = True
) -> str:
    """
    Create a subdirectory within the project directory and print information.

    This function creates a subdirectory at the specified relative path within the
    project directory. It also provides information about the created subdirectory
    and obfuscates sensitive user-specific information in the output.

    Parameters:
    -----------
    project_path : str
        The absolute path to the project directory.
    rel_path : str, optional
        The relative path to the subdirectory to be created within the project
        directory. Default is an empty string.
    verbose : bool, optional
        If True, print information about the created subdirectory. Default is True.

    Returns:
    --------
    str
        The absolute path to the created subdirectory.

    Notes:
    ------
    The function replaces the absolute path to the project directory with
    '[project_dir]' to protect sensitive user-specific information that may be
    present in the path.

    Examples:
    ---------
    >>> create_subdir('/path/to/project', 'data', verbose=True)
    [project_dir]/data exists
    ✔ [project_dir]/data created.
    '/path/to/project/data'

    >>> create_subdir('/path/to/project', 'results', verbose=False)
    '/path/to/project/results'
    """
    path = os.path.join(project_path, rel_path)
    if verbose:
        print_path_info(path)
    if not os.path.exists(path):
        os.makedirs(path)
        project_dir = get_project_dir()
        print(bold('✔ ' + path.replace(project_dir, '[project_dir]')), 'created.')
    return path


def commented_return(s: bool, o: str, a: str, *args: str) -> tuple:
    """
    Print a status message and return a tuple of additional arguments.

    This function prints a status message based on the value of `s` (True or
    False), and then returns a tuple containing the provided additional arguments.

    Parameters:
    -----------
    s : bool
        The status indicator. If True, the message will be green with a '✔' (checkmark);
        if False, it will be red with a '✘' (cross).
    o : str
        The message or object description.
    a : str
        The additional message or object description.
    *args : str
        Additional arguments to be included in the returned tuple.

    Returns:
    --------
    tuple
        A tuple containing the additional arguments passed to the function.

    Examples:
    ---------
    >>> commented_return(True, 'Operation', 'Completed', 'Additional', 'Arguments')
    Operation Completed Additional Arguments
    ('Additional', 'Arguments')

    >>> commented_return(False, 'Task', 'Failed')
    Task Failed
    ()
    """
    status = lambda s, o: bold(green(f'✔ {o}') if s else red(f'✘ {o}'))
    print(status(s, o), a)
    return args


# Multi-indexing utils
def _load_struct(no_comment: bool = True) -> pd.DataFrame:
    """
    Load a dataset schema description from the "struct.json" file.

    This function reads the schema description from the "struct.json" file
    located in the dataset directory. The schema is typically used to guide
    type casting and optimization during dataset loading.

    Parameters:
    -----------
    no_comment : bool, optional (default=True)
        If True, the function will load the schema description without
        generating any status comments. If False, it will print a status
        message indicating that the schema has been loaded.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the dataset schema description.

    Examples:
    ---------
    >>> _load_struct()
    # Loads the dataset schema description silently and returns it as a DataFrame.

    >>> _load_struct(no_comment=False)
    # Loads the dataset schema description with a status message and returns it as a DataFrame.
    """
    dataset_dir = get_dataset_dir()
    data = pd.read_json(
        os.path.join(dataset_dir, "struct.json"),
        typ="frame",
        orient="index"
    )
    # print(bold('✔ struct'), 'loaded')
    return data if no_comment else commented_return(True, "struct", "loaded", data)


_DATASET_SCHEMA = _load_struct()


def get_element_by_id(id: str, label: str) -> str:
    """
    Get an element from the dataset schema description by its ID and label.

    Parameters:
    -----------
    id : str
        The ID of the element.
    label : str
        The label (e.g., 'group', 'subgroup', 'domain', etc.) to retrieve.

    Returns:
    --------
    str
        The value of the element specified by the ID and label.

    Examples:
    ---------
    >>> get_element_by_id('example_id', 'group')
    # Retrieves the 'group' value associated with 'example_id'.
    """
    return _DATASET_SCHEMA.loc[_DATASET_SCHEMA.id == id, label].values[0]

"""
get_group = lambda id: get_element_by_id(id, "group")
get_subgroup = lambda id: get_element_by_id(id, "subgroup")
get_domain = lambda id: get_element_by_id(id, "domain")
get_format = lambda id: get_element_by_id(id, "format")
get_unity = lambda id: get_element_by_id(id, "unity")
get_astype = lambda id: get_element_by_id(id, "astype")
get_nan_code = lambda id: get_element_by_id(id, "nan_code")
get_nap_code = lambda id: get_element_by_id(id, "nap_code")
"""

# get columns labels from ancestor
# _get_labels = lambda k, v: _DATASET_SCHEMA.name[_DATASET_SCHEMA[k] == v].values

def get_labels(col: str, val: Any) -> np.ndarray:
    """
    Get column labels based on the specified ancestor.

    Parameters:
    -----------
    col : str
        The column name.
    val : Any
        The value (e.g., 'group', 'subgroup', 'domain', etc.) to retrieve labels for.

    Returns:
    --------
    np.ndarray
        An array containing the labels corresponding to the specified ancestor.

    Examples:
    ---------
    >>> get_labels('group', 'example_group')
    # Retrieves labels associated with the 'example_group'.
    """
    return _DATASET_SCHEMA.name[_DATASET_SCHEMA[col] == val].to_numpy()


# get_group_labels = lambda gp_label: get_labels('group', gp_label)


def new_multi_index(levels: Optional[List[str]] = None) -> pd.MultiIndex:
    """
    Create a new MultiIndex based on specified levels.

    Parameters:
    -----------
    levels : list of str, optional
        A list of levels to include in the MultiIndex. Default is ['group'].

    Returns:
    --------
    pd.MultiIndex
        A MultiIndex constructed from the specified levels.

    Examples:
    ---------
    >>> new_multi_index(levels=['group', 'subgroup'])
    # Creates a new MultiIndex with levels 'group' and 'subgroup'.
    """
    if levels is None:
        levels = ["group"]
    return pd.MultiIndex.from_frame(
        _DATASET_SCHEMA[levels + ["name"]],
        names=levels+["var"]
    )
