from IPython.display import display
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


""" Align utils
"""

def align_df2_on_df1(
    pk_name: str,
    df1: pd.DataFrame,
    df2: pd.DataFrame
) -> pd.DataFrame:
    """Aligns the DataFrame `df2` on the primary key column `pk_name` of
    DataFrame `df1`. Returns a DataFrame `aligned_v2` where the order of rows
    of `df2` is changed according to the primary key column of `df1`.
    
    Parameters
    ----------
    pk_name : str
        The name of the primary key column that will be used for alignment
    df1 : pd.DataFrame
        The DataFrame to align `df2` with
    df2 : pd.DataFrame
        The DataFrame to be aligned on `df1`
    
    Returns
    -------
    A DataFrame with the same columns as `df2`, but with rows aligned on `df1`
    """
    df1_pk = df1[pk_name]  # idx (v1) -> pk map
    df2_pk = df2[pk_name]  # idx (v2) -> pk map

    # Create a mapping from pk to index in df2
    # pk -> idx (in df2) reverse map
    pk_to_df2_idx = (
        df2_pk.reset_index()
        .set_index("SK_ID_CURR")["index"]  # important : don't remove [""]!!
    )

    # Create a mapping from pk in df1 to index in df2
    aligned_v2_index = df1_pk.map(pk_to_df2_idx)

    # Return df2 aligned on df1 based on the mapping
    return df2.iloc[aligned_v2_index]


""" Diff utils
"""

def df_eq(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    try:
        return (df1.isnull() & df1.isnull()) | (df1 == df2)
    except ValueError as e:
        print("DataFrames cannot be compared.")
        print(f"Caught ValueError: {e}")
        return pd.DataFrame([False])


def df_neq(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    return ~df_eq(df1, df2) 


def check_dtypes_alignment(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    try:
        dtypes_diff = df1.dtypes != df2.dtypes
        if (~dtypes_diff).all():
            print("dtypes are aligned")
        else:
            print("dtypes diffs:")
            display(df1.dtypes[dtypes_diff])
            display(df2.dtypes[dtypes_diff])
    except ValueError as e:
        print("DataFrames cannot be compared.")
        print(f"Caught ValueError: {e}")


def replace_chars(s1: pd.Series, s2: pd.Series, default="") -> pd.Series:
    """Removes occurrences of s1 from s2."""
    s1 = s1.fillna(default)
    s2 = s2.fillna(default)
    return (pd.Series([
        x2.replace(x1, "")
        if x1 != x2 else ""
        for x1, x2 in zip(s1, s2)
    ], index=s1.index))


def micro_test_replace_char():
    s1 = pd.Series(["A", "B", "C", None, None])
    s2 = pd.Series(["A", "D", "C", "X", None])
    print(s2.where(s1 == s2, s2.str.cat(s1, sep="")))
    print(replace_chars(s1, s2))


def safe_diff_series(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """Subtracts the values in s1 from the values in s2.
    If the columns are str, return values of s2 with s1 one removed.
    """
    if (
        pd.api.types.is_numeric_dtype(s1)
        and pd.api.types.is_numeric_dtype(s2)
    ):
        return s2 - s1
    else:
        return replace_chars(s1, s2)


def safe_diff_dataframe(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Subtracts the values in df1 from the values in df2.
    If the columns are not numeric, return values of s2 with s1 one removed.
    """
    return df1.apply(lambda s: safe_diff_series(s, df2[s.name]))
