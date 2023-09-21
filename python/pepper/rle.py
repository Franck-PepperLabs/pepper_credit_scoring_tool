from typing import Union

import itertools
import pandas as pd
import numpy as np


def row_rle(row: pd.Series) -> np.ndarray:
    """
    Reduce consecutive elements in a row to value and count pairs using
    RLE.

    DEPRECATED use `series_rle_reduction` instead
    
    Parameters
    ----------
    row : pandas Series
        A row of data, represented as a tuple or 1D ndarray.
    
    Returns
    -------
    ndarray
        An array of value and count pairs.
    """
    # Replace np.nan by "☗" and cast row in ndarray
    row = row.astype(object)
    row.fillna("☗", inplace=True)
    row_a = row.to_numpy() 

    # Find the locations of elements that are not equal to the previous one
    # The additional `True` marks the end of sequence
    diff = np.concatenate(([True], row_a[:-1] != row_a[1:], [True]))

    # Calculate the count of each unique value using the diffs array
    lens = np.diff(np.where(diff)[0])

    # Extract the unique values from the row, including null values
    vals = row_a[diff[:-1]]

    # Replace "☗" by np.nan
    vals[vals == "☗"] = np.nan

    # Create and return an array of value and count pairs
    return np.column_stack((vals, lens))


def series_rle_reduction(
    s: pd.Series,
    # TODO extension : dtype = None : pour forcer un astype en supposons que le type peut
    # avoir été upcasté par exemple par un groupby
    # TODO extension : decimals: int = None : pour permettre de regrouper des termes proches
) -> np.ndarray:
    """
    Reduce consecutive elements in a pandas Series to value and count pairs using
    Run-Length Encoding (RLE).

    Parameters
    ----------
    s : pd.Series
        The pandas Series to be processed.

    Returns
    -------
    np.ndarray
        An array of value and count pairs, where each pair consists of a unique
        value from the input Series and the count of consecutive occurrences of
        that value.

    Notes
    -----
    This function replaces consecutive occurrences of the same value in the input
    Series with a single entry that contains the value and the count of
    consecutive occurrences. It is commonly used for compressing sequences of
    repeated values.

    Examples
    --------
    >>> import pandas as pd
    >>> from pepper.feat_eng import series_rle_reduction
    >>> s = pd.Series([1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4])
    >>> result = series_rle_reduction(s)
    >>> print(result)
    [(1, 2) (2, 3) (3, 2) (4, 4)]

    >>> s = pd.Series([None, None, 'A', 'A', None, None, 'B'])
    >>> result = series_rle_reduction(s)
    >>> print(result)
    [(nan, 2) ('A', 2) (nan, 2) ('B', 1)]

    The function replaces consecutive identical values with a single entry
    containing the value and the count of consecutive occurrences.
    """
    # Avoid doing this: grouped.apply(lambda df: df.apply(rle_reduction))
    # Instead, use this: grouped.apply(data_rle_reduction)

    # Ensure safety if 's' is None or empty; return an empty array
    if s is None or s.shape[0] == 0:
        return np.array([])

    # Replace np.nan by "☗" and cast the Series to a numpy ndarray
    s = s.astype(object)
    s.fillna("☗", inplace=True)
    a = s.to_numpy()
    # print("input arr")
    # display(a)

    # Find the locations of elements that are not equal to the previous one
    # The additional `True` marks the start and the end of sequence
    diff = np.concatenate(([True], a[:-1] != a[1:], [True]))
    # print("diff arr")
    # display(diff)

    # print("where(diff)")
    # display(np.where(diff))
    # display(np.where(diff)[0])

    # Calculate the count of each unique value using the diffs array
    lens = np.diff(np.where(diff)[0])
    # print("lens arr")
    # display(lens)

    # Extract the unique values from the row, including null values
    vals = a[diff[:-1]]
    # print("vals arr")
    # display(vals)

    # Replace "☗" by np.nan
    vals[vals == "☗"] = np.nan

    # Create and return an array of value and count pairs
    return tuple(zip(vals, lens)) #np.column_stack((vals, lens))


def data_rle_reduction(data: pd.DataFrame) -> pd.Series:
    """
    Reduce consecutive elements in each column of a DataFrame to value and count pairs using RLE (Run-Length Encoding).

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to process.

    Returns
    -------
    pd.Series
        A Series containing a sequence of tuples for each column. Each tuple consists of a unique value and its corresponding count in the column.

    Notes
    -----
    - Avoid using this function within nested apply operations, as it may lead to performance issues.
    - Instead, apply it directly to the DataFrame you want to process.
    - Missing values in the DataFrame will be temporarily replaced with the symbol "☗" during processing.

    Examples
    --------
    >>> import pandas as pd
    >>> from pepper.feat_eng import data_rle_reduction
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 2, 2, 1],
    ...     'B': [None, 1, 1, None, 2]
    ... })
    >>> result = data_rle_reduction(df)
    >>> print(result)
    A    ((1, 1), (2, 3), (1, 1))
    B           ((☗, 1), (1, 2), (☗, 2), (2, 1))
    dtype: object
    """
    # Avoid doing this: grouped.apply(lambda df: df.apply(rle_reduction))
    # Instead, use this: grouped.apply(data_rle_reduction)

    # Replace np.nan by "☗" and cast row in ndarray
    data = data.astype(object)
    data.fillna("☗", inplace=True)
    a = data.T.to_numpy()
    # print("input mx")
    # display(a)

    # Find the locations of elements that are not equal to the previous one
    # The additional `True` marks the end of sequence
    diff = np.concatenate(
        [
            np.ones((a.shape[0], 1), dtype=bool),
            a[:, :-1] != a[:, 1:],
            np.ones((a.shape[0], 1), dtype=bool)
        ],
        axis=1
    )
    # print("diff mx")
    # display(diff)

    # Calculate the count of each unique value using the diffs array
    # print("where(diff)")
    # display(np.where(diff))
    # display(np.where(diff)[1])
    # display(np.diff(np.where(diff)[1]))

    # Note : the concatenated sequences are separated by -n where n = data.shape[0]
    lensep = np.diff(np.where(diff)[1])  #.reshape(-1, data.shape[1])
    # print("lens mx", len(lensep))
    # display(lensep)

    # Extract the unique values from the df, including null values
    # Note : here, the sequences are concatened without separator
    vals = a[diff[:, :-1]]
    # print("vals mx", len(vals))
    # display(vals)

    # print("val lens array")
    lens = lensep[lensep > 0]
    # display(lens)

    # print("separators pos")
    seps = np.where(lensep < 0)[0]
    # il faut retrancher 1 pour chaque pos qui précède
    # en d'autres termes, retrancher son index à chaque pos
    seps = seps - np.arange(seps.shape[0])
    seps = seps.tolist()
    # display(seps)

    # Replace "☗" by np.nan
    vals[vals == "☗"] = np.nan
    # display(vals)
    # display(vals.shape)

    # Create and return a per column tuple of tuples of value and count pairs
    starts = [0] + seps
    ends = seps + [lensep.shape[0]]
    # print("starts:", starts)
    # print("ends:", ends)

    # Soluce 1 (timeit to compare)
    """return pd.Series(
        tuple(c for c in zip(vals[s:e], lens[s:e]))
        for s, e in zip(starts, ends)
    )"""
    # Soluce 2 (timeit to compare)
    return pd.Series(
        (
            tuple(itertools.islice(zip(vals, lens), s, e))
            for s, e in zip(starts, ends)
        )
        #,index=data.columns 
    )


def jumps_rle(s: pd.Series) -> tuple:
    """
    Encode a series using run-length encoding (RLE).
    
    RLE is a lossless data compression algorithm that works by
    reducing the amount of data needed to represent a repeating sequence
    of values. This function takes a pandas series and returns a tuple
    containing the encoded sequence.
    
    Parameters
    ----------
    s : pd.Series
        A pandas series containing the data to be encoded.
    
    Returns
    -------
    tuple
        A tuple containing the encoded sequence.
    
    Notes
    -----
        - A group is non-empty, and it is therefore a slice of a series. 
        - It is assumed there are no NaNs in the input series. 
        - However, if `s` has only one element, the `diff` of `s` has one less element.
    
    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4, 5, 6])
    >>> print(jumps_rle(s))
    ((1, 5),)
    >>> s = pd.Series([1, 1, 1, 2, 2, 3])
    >>> print(jumps_rle(s))
    ((0, 2), (1, 1), (0, 1), (1, 1))
    >>> gpby.agg([min, max, len, jumps_rle])
    """

    if s.shape[0] == 1:
        return ()
    # Extract s values ndarray
    a = s.to_numpy()
    # Calculate jumps between values
    j = np.diff(a, prepend=-1)
    # Find the locations of jumps that differ from the previous one
    # The additional `True` marks the end of sequence
    diff = np.concatenate(([True], j[:-1] != j[1:], [True]))
    # Calculate the count of each unique value using the diffs array
    lens = np.diff(np.where(diff)[0])
    # Extract the unique values from the row
    vals = j[diff[:-1]]
    return tuple(zip(vals, lens))


""" RLE expand
"""


def rle_expr_to_numpy(
    rle_expr: Union[list, tuple, np.ndarray]
) -> np.array:
    # Convert inputs to NumPy arrays if they are not already
    return rle_expr if isinstance(rle_expr, np.ndarray) else np.array(rle_expr)


def rle_support_size(rle: np.array):
    return np.sum(rle[:, 1])


def rle_expand_expr(
    rle_expr: Union[list, tuple, np.ndarray]
) -> np.ndarray:
    """
    Expand a series from its RLE (Run-Length Encoding) representation,
    which is given as a sequence of value-count pairs.

    Parameters
    ----------
    rle_expr : Union[list, tuple, np.ndarray]
        The RLE representation of the series,
        typically obtained from series_rle_reduction.

    Returns
    -------
    np.ndarray
        The original series as a NumPy array.
        
    Example
    -------
    >>> rle_expand(((0, 1), (1, 2), (2, 3)))
    array([0, 1, 1, 2, 2, 2])
    >>> rle_expand([[1, 1], [0, 2], [.5, 1]])
    array([1. , 0. , 0. , 0.5])
    >>> rle_series.apply(rle_expand_expr)
    
    Notes
    -----
    This function is essentially a wrapper for the Numpy `repeat` function.
    
    Raises
    ------
    ValueError
        If the input format is not valid, i.e.,
        if it doesn't consist of value-count pairs.
    """
    # Convert input to a NumPy array if it's not already
    # if not isinstance(rle_expr, np.ndarray):
    #     rle = np.array(rle_expr)

    rle = rle_expr_to_numpy(rle_expr)

    if rle.size == 0:
        return np.array([])

    # Extract values and counts
    if rle.shape[1] == 2:
        values, counts = rle[:, 0], rle[:, 1].astype(int)
    else:
        raise ValueError("Invalid input format. Input must consist of value-count pairs.")

    # Expand the series
    expanded_series = np.repeat(values, counts)

    # Determine the dtype based on the type of the first element
    dtype = type(values[0])

    return expanded_series.astype(dtype)


def support_max_indice(
    rle_support_expr: Union[list, tuple, np.ndarray]
) -> int:
    # Convert input to a NumPy array if it's not already
    if not isinstance(rle_support_expr, np.ndarray):
        rle_support = np.array(rle_support_expr)

    if rle_support.size == 0:
        return -1

    # Extract values and counts
    if rle_support.shape[1] == 2:
        jumps, counts = rle_support[:, 0], rle_support[:, 1]
    else:
        raise ValueError("Invalid input format. Input must consist of jump-count pairs.")
    
    return -1 + np.sum(jumps * counts)