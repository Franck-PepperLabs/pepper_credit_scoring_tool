from typing import Union, Any, List, Tuple

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
        data.isnull()
        .replace({True: na_symb, False: nona_symb})
        .apply(lambda r: "".join(r), axis=1)
    )


def row_rle(row: pd.Series) -> np.ndarray:
    """Reduces consecutive elements in a row to value and count pairs using
    RLE.
    
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
    """true_dtype = row.dtype
    tmp_dtype = true_dtype
    if row.hasnans:
        if row.dtype in [int, float]:
            row = row.astype(object)
            tmp_dtype = object
        row.fillna("☗", inplace=True)"""
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
    """if row.dtype == object:
        vals[vals == "☗"] = np.nan
    else:
        vals[vals == -1] = np.nan"""

    # Create and return an array of value and count pairs
    return np.column_stack((vals, lens))


# Experimental 
def vect_row_rle(*rows: List[pd.Series]) -> List[np.ndarray]:
    """Reduces consecutive elements in a row to value and count pairs using
    RLE.

    Parameters
    ----------
    *rows : list of pandas Series
        A list of rows of data, each represented as a pd.Series.

    Returns
    -------
    List of ndarrays
        A list of arrays of value and count pairs for each input row.
    """
    # Make sure all rows have the same length
    row_length = len(rows[0])
    assert_msg = "All rows must have the same length"
    assert all(len(row) == row_length for row in rows), assert_msg

    # Replace np.nan by "☗" and cast rows in ndarray
    rows_with_fillna = [row for row in rows if hasattr(row, "fillna")]
    rows_a = np.asarray(
        [
            row_with_fillna.fillna("☗").to_numpy()
            for row_with_fillna in rows_with_fillna
        ]
    )

    # Run-length encode the rows
    diff = np.diff(rows_a, axis=1, prepend=1)
    lens = np.diff(np.where(diff)[1], axis=1)
    vals = rows_a[:, :-1][diff != 0]
    rle = np.column_stack((vals, lens))

    # Convert RLE to list of column stacks
    num_rows = len(rows)
    result = [None] * num_rows
    current_idx = 0
    for i, row in enumerate(rows):
        if hasattr(row, "fillna"):
            result[i] = rle[current_idx : current_idx + row_length]
            current_idx += row_length
        else:
            result[i] = None
    return result

