import pandas as pd

from pepper.np_utils import subindex as npsi


def subindex(s: pd.Series, sorted: bool = False) -> pd.DataFrame:
    """Subindexes the elements in a Pandas Series and returns a DataFrame with
    the original Series and the subindexed values.

    Parameters
    ----------
    s : pandas.Series
        The input Series to subindex.
    sorted : bool, optional
        Whether the input array is sorted or not. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the original input Series and the subindexed
        values.

    Raises
    ------
    ValueError
        If the input Series is not one-dimensional.

    Examples
    --------
    >>> s = pd.Series([0, 0, 1, 1, 1, 3, 5, 5, 11, 11, 11, 11], name='my_series')
    >>> subindex(s)
       my_series  subindex
    0          0         0
    1          0         1
    2          1         0
    3          1         1
    4          1         2
    5          3         0
    6          5         0
    7          5         1
    8         11         0
    9         11         1
    10        11         2
    11        11         3

    Notes
    -----
    Setting the 'sorted' parameter to True can significantly improve
    performance, but will produce incorrect results if the input Series is
    not sorted.

    The subindex operation assigns a unique integer identifier to each element
    based on the number of times the element has occurred previously in the
    array.
    """
    # Check that the input Series is one-dimensional
    if s.ndim != 1:
        raise ValueError("input Series must be one-dimensional")

    # Apply subindexing operation to the values of the input Series
    s_sub = pd.Series(
        npsi(s.values, sorted),
        index=s.index, name="subindex"
    )

    # Concatenate the input Series and the subindexed values into a DataFrame
    return pd.concat([s, s_sub], axis=1)
