from typing import Union, Any, List, Callable

import pandas as pd
import numpy as np

from pepper.univar import agg_value_counts


""" Missing values
"""

def nullify(
    data: Union[pd.Series, pd.DataFrame],
    val: Union[Any, List[Any]]
) -> None:
    """
    Replace occurrences of the value(s) val by None in a pandas Series or
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
    """
    Reduce the long tail of a pandas Series by replacing infrequent values
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
    na_symbol: str = "◻",
    nona_symbol: str = "◼"
) -> pd.Series:
    """
    Return a Series containing a profile of the missing values in each row
    of a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to get the missing value profile for.
    na_symbol : str
        The symbol to use for missing values.
    nona_symbol : str
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
        .replace({True: na_symbol, False: nona_symbol})
        # Concatenate the values in each row to form a single string
        .apply(lambda r: "".join(r), axis=1)
    )


# Generalization of the previous
def get_profile(
    data: pd.DataFrame,
    predicate: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
    na_symbol: str = "◻",
    nona_symbol: str = "◼"
) -> pd.Series:
    """
    Return a Series containing a profile of the values in each row of a
    DataFrame after applying a given predicate function to the input DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to profile.
    predicate : Callable[[pd.DataFrame], pd.DataFrame], optional
        The predicate function to apply to each row of the DataFrame.
        Default is the identity function, which returns the original DataFrame.
    na_symbol : str, optional
        The symbol to use for missing values, by default "◻".
    nona_symbol : str, optional
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
        .replace({True: na_symbol, False: nona_symbol})
        # Concatenate the values in each row to form a single string
        .apply(lambda r: "".join(r), axis=1)
    )

