import numpy as np


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
