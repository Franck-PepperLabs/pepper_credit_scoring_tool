from typing import Union, Any, List, Tuple, Callable

import itertools

import pandas as pd
import numpy as np

from pepper.univar import agg_value_counts


""" Missing values
"""

def nullify(
    data: Union[pd.Series, pd.DataFrame],
    val: Union[Any, List[Any]]
) -> None:
    """Replaces occurrences of the value(s) val by None in a pandas Series or
    DataFrame.
    
    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        The data to be modified.
    val : any value or list of values
        The value(s) to replace by None.
    
    Returns
    -------
    None
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data.replace(val, np.nan, inplace=True)
    else:
        raise TypeError("Input data must be a pandas Series or DataFrame.")


""" Long tail
"""


def reduce_long_tail(
    s: pd.Series,
    agg: Union[None, bool, float, int] = .01
) -> pd.Series:
    """Reduces the long tail of a pandas Series by replacing infrequent values
    with a single category.

    Parameters
    ----------
    s : pd.Series
        The pandas Series to be processed.
    agg : Union[None, bool, float, int], optional
        An optional parameter that specifies how to aggregate the results.
        If None, returns the raw value counts and relative frequencies.
        If a float between 0 and 1, returns a DataFrame with a row containing
        all values whose cumulative proportion of occurrences is less than agg,
        with the remaining values aggregated. If True, determines the threshold
        automatically using the first index at which the proportion is less
        than the sum of the next proportions.
    
    Returns
    -------
    pd.Series
        A pandas Series with the same index as `s`, where values that occur
        infrequently have been replaced with a single category represented by a
        string of the form "value1:value2", where `value1` is the first and
        `value2` is the last value in the long tail. All other values remain
        unchanged.

    Notes
    -----
    The long tail is defined as all values in `s` except for the most frequent
    `k-1` values, where `k` is the number of unique values in `s`.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pepper.feat_eng import reduce_long_tail
    >>> s = pd.Series([np.nan, 1, 2, 1, 2, 3, 1, 1, np.nan])
    >>> print(list(reduce_long_tail(s)))
    [nan, 1.0, '2.0:3.0', 1.0, '2.0:3.0', '2.0:3.0', 1.0, 1.0, nan]
    >>> print(list(reduce_long_tail(s.dropna().astype(int))))
    [1, '2:3', 1, '2:3', '2:3', 1, 1]
    """
    avc = agg_value_counts(s, agg=agg)
    red_s = s.copy()
    # in_head = s.notna() & s.isin(avc.index[:-1])
    # red_s[in_head] = s[in_head].apply(infer_type(s))
    in_long_tail = s.notna() & ~s.isin(avc.index[:-1])
    red_s[in_long_tail] = avc.index[-1]
    return red_s


""" NA profiling
"""

def get_na_profile(
    data: pd.DataFrame,
    na_symb: str = "◻",
    nona_symb: str = "◼"
) -> pd.Series:
    """Returns a Series containing a profile of the missing values in each row
    of a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to get the missing value profile for.
    na_symb : str
        The symbol to use for missing values.
    nona_symb : str
        The symbol to use for non-missing values.

    Returns
    -------
    pd.Series
        A Series containing a profile of the missing values in each row of the
        input DataFrame.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({  # create a dataframe with some missing values
    >>>     'A': [1, 2, 3, None, 5],
    >>>     'B': [None, 7, None, None, 11],
    >>>     'C': [13, 14, None, 16, None],
    >>> })
    >>> na_profile = get_na_profile(df)  # get the missing value profile
    >>> print(na_profile)  # display the profile
    0    ◼◻◼
    1    ◻◼◻
    2    ◼◻◻
    3    ◻◻◻
    4    ◼◻◻
    dtype: object
    """
    return (
        # Apply the predicate function
        data.isnull()
        # Replace NaN values with the given symbol 
        .replace({True: na_symb, False: nona_symb})
        # Concatenate the values in each row to form a single string
        .apply(lambda r: "".join(r), axis=1)
    )


# Generalisation of the previous
def get_profile(
    data: pd.DataFrame,
    predicate: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
    na_symb: str = "◻",
    nona_symb: str = "◼"
) -> pd.Series:
    """Returns a Series containing a profile of the values in each row of a
    DataFrame after applying a given predicate function to the input DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to profile.
    predicate : Callable[[pd.DataFrame], pd.DataFrame], optional
        The predicate function to apply to each row of the DataFrame.
        Default is the identity function, which returns the original DataFrame.
    na_symb : str, optional
        The symbol to use for missing values, by default "◻".
    nona_symb : str, optional
        The symbol to use for non-missing values, by default "◼".

    Returns
    -------
    pd.Series
        A Series containing a profile of the values in each row of the input
        DataFrame after applying the predicate function.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, None, 5],
    ...     'B': [None, 7, None, None, 11],
    ...     'C': [13, 14, None, 16, None],
    ... })
    >>> # get the missing value profile using `isnull` as the predicate function
    >>> na_profile = get_profile(df, lambda x: x.isnull())
    >>> print(na_profile)  # display the profile
    0    ◼◻◼
    1    ◻◼◻
    2    ◼◻◻
    3    ◻◻◼
    4    ◼◻◼
    dtype: object
    """
    return (
        # Apply the predicate function
        predicate(data) 
        # Replace NaN values with the given symbol 
        .replace({True: na_symb, False: nona_symb})
        # Concatenate the values in each row to form a single string
        .apply(lambda r: "".join(r), axis=1)
    )


def row_rle(row: pd.Series) -> np.ndarray:
    """Reduces consecutive elements in a row to value and count pairs using
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


def series_rle_reduction(s: pd.Series) -> np.ndarray:
    # don't do this : z = gpby.apply(lambda df: df.apply(rle_reduction))
    # but this : zz = gpby.apply(data_rle_reduction)
    # Replace np.nan by "☗" and cast row in ndarray

    # TODO : sécuriser si s est None ou vide : retourner un tableau vide
    if s is None or s.shape[0] == 0:
        return np.array([])

    s = s.astype(object)
    s.fillna("☗", inplace=True)
    a = s.to_numpy()
    # print("input arr")
    # display(a)

    # Find the locations of elements that are not equal to the previous one
    # The additional `True` marks the end of sequence
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
    # don't do this : z = gpby.apply(lambda df: df.apply(rle_reduction)) [10 m 30]
    # but this : zz = gpby.apply(data_rle_reduction) [3 min 20]
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
    """Encodes a series using run-length encoding (RLE).
    
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
        >>> s = pd.Series([1, 1, 1, 2, 2, 3])
        >>> jumps_rle(s)
        ((1, 3), (2, 2), (3, 1))
        >>> gpby.agg([min, max, len, jumps_rle])
    """

    if s.shape[0] == 1:
        return ()
    # Extract s values ndarray
    a = s.to_numpy()
    # Calculate jumps between values
    j = np.diff(a)
    # Find the locations of jumps that differ from the previous one
    # The additional `True` marks the end of sequence
    diff = np.concatenate(([True], j[:-1] != j[1:], [True]))
    # Calculate the count of each unique value using the diffs array
    lens = np.diff(np.where(diff)[0])
    # Extract the unique values from the row
    vals = j[diff[:-1]]
    return tuple(zip(vals, lens))
