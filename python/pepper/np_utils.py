# pepper/np_utils.py
import numpy as np
from typing import Any, List, Union

def subindex(
    a: np.ndarray,
    sorted: bool = False
) -> np.ndarray:
    """
    Return an array of the same shape as 'a', where each element is
    assigned a unique integer identifier based on the number of times the
    element has occurred previously in the array.

    Parameters
    ----------
    a : numpy.ndarray
        The array to subindex.
    sorted : bool, optional
        Whether the input array is sorted or not. Defaults to False.

    Returns
    -------
    numpy.ndarray
        The subindex array.

    Raises
    ------
    ValueError
        If the input array is not one-dimensional.
    
    Example
    -------
    >>> a = np.array([0, 0, 1, 1, 1, 3, 5, 5, 11, 11, 11, 11])
    >>> subindex(a)
    array([0, 1, 0, 1, 2, 0, 0, 1, 0, 1, 2, 3])
    >>> a = np.array([1, 11, 5, 11, 1, 11, 5, 0, 3, 11, 0, 1])
    >>> subindex(a)
    array([0, 0, 0, 1, 1, 2, 1, 0, 0, 3, 1, 2])
    
    Notes
    -----
    Setting the 'sorted' parameter to True can significantly improve
    performance, but will produce incorrect results if the input array is
    not sorted.
    """
    if len(a.shape) != 1:
        raise ValueError("input array must be one-dimensional")

    # Check if input array is already sorted
    idx = a
    if not sorted:
        idx = np.argsort(a)
        a = a[idx]

    # Find unique values, their indices, and their counts
    _, i, c = np.unique(a, return_index=True, return_counts=True)

    # Apply subindex operation
    x = np.zeros_like(a)
    for k in range(1, np.max(c)):
        x[i[c > k] + k] = k

    if sorted:
        return x

    # Apply inverse permutation if input array was sorted
    inv_idx = np.empty_like(idx)
    inv_idx[idx] = np.arange(len(idx))
    return x[inv_idx]


def subindex_nd(
    a: np.ndarray,
    sorted: bool = False
) -> np.ndarray:
    """
    Return an array of the same shape as 'a', where each element is
    assigned a unique integer identifier based on the number of times the
    element has occurred previously in the array.

    Parameters
    ----------
    a : numpy.ndarray
        The array to subindex.
    sorted : bool, optional
        Whether the input array is sorted or not. Defaults to False.

    Returns
    -------
    numpy.ndarray
        The subindex array.

    Raises
    ------
    ValueError
        If the input array is not two-dimensional.

    Example
    -------
    >>> a = np.array([[0, 1], [0, 1], [1, 2], [1, 2], [1, 2]])
    >>> subindex_nd(a)
    array([0, 1, 0, 1, 2])

    Notes
    -----
    Setting the 'sorted' parameter to True can significantly improve
    performance, but will produce incorrect results if the input array is
    not sorted.
    """
    if len(a.shape) != 2:
        raise ValueError("Input array must be two-dimensional")

    # Check if input array is already sorted
    idx = a
    if not sorted:
        idx = np.lexsort(a.T)  # Sort by columns
        a = a[idx]

    # Find unique rows, their indices, and their counts
    _, i, c = np.unique(a, axis=0, return_index=True, return_counts=True)

    # Apply subindex operation
    x = np.zeros_like(a[:, 0], dtype=int)
    for k in range(1, np.max(c)):
        x[i[c > k] + k] = k

    if sorted:
        return x

    # Apply inverse permutation if input array was sorted
    inv_idx = np.empty_like(idx)
    inv_idx[idx] = np.arange(len(idx))
    
    return x[inv_idx]


def ndarray_to_list(arr: np.ndarray) -> Union[List[Any], Any]:
    """
    Recursively convert a numpy ndarray to an equivalent nested list.

    Parameters
    ----------
    arr : np.ndarray
        The numpy ndarray to be converted.

    Returns
    -------
    Union[List[Any], Any]
        The converted list. If the input is not an ndarray, it is returned as is.
        If the input is a one-dimensional ndarray, it is converted to a regular list.
        If the input is a multi-dimensional ndarray, it is converted to a nested list structure.

    Examples
    --------
    >>> import numpy as np
    >>> s = [[1, 2], [3, 4]]
    >>> a = np.array(s)
    >>> converted_list = ndarray_to_list(a)
    >>> print(converted_list)
    [[1, 2], [3, 4]]

    >>> b = np.array([1, 2, 3])
    >>> converted_list = ndarray_to_list(b)
    >>> print(converted_list)
    [1, 2, 3]

    >>> c = 42
    >>> converted_value = ndarray_to_list(c)
    >>> print(converted_value)
    42
    """
    if not isinstance(arr, np.ndarray):
        return arr
    if arr.ndim == 1:
        return arr.tolist()
    return [ndarray_to_list(sub_arr) for sub_arr in arr]


def reconstruct_ndarray(arr: list | tuple | np.ndarray) -> np.ndarray:
    """
    Recursively reconstructs a multidimensional ndarray from a structure
    of nested ndarrays, lists, and/or tuples.

    Parameters
    ----------
    arr : np.ndarray or list or tuple
        The input structure to be reconstructed into an ndarray.

    Returns
    -------
    np.ndarray
        The reconstructed ndarray.

    Notes
    -----
    This function takes a structure consisting of nested ndarrays, lists,
    and/or tuples, and reconstructs it into a multidimensional ndarray
    by stacking the arrays along the appropriate dimension.
    If the input is already an ndarray, it is returned as is.

    Examples
    --------
    >>> import numpy as np
    >>> def reconstruct_ndarray(arr):
    ...     if not isinstance(arr, (list, tuple, np.ndarray)):
    ...         return arr
    ...     return np.stack([reconstruct_ndarray(sub_arr) for sub_arr in arr])
    ...
    >>> x = [
    ...     np.array([(1, 2, 3), (4, 5, 6)]),
    ...     np.array([(7, 8, 9), (10, 11, 12)])
    ... ]
    >>> print(x)
    [array([[1, 2, 3],
            [4, 5, 6]]),
     array([[ 7,  8,  9],
            [10, 11, 12]])]
    >>> reconstructed_x = reconstruct_ndarray(x)
    >>> print(reconstructed_x)
    array([[[ 1,  2,  3],
            [ 4,  5,  6]],

           [[ 7,  8,  9],
            [10, 11, 12]]])
    """
    if not isinstance(arr, (list, tuple, np.ndarray)) or len(arr) == 0:
        return arr
    return np.stack([reconstruct_ndarray(sub_arr) for sub_arr in arr])
