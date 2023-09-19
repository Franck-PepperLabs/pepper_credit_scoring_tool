from IPython.display import display
import pandas as pd
from pepper import np_utils


def subindex(s: pd.Series, sorted: bool = False) -> pd.DataFrame:
    """
    Subindex the elements in a Pandas Series and returns a DataFrame with
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
        np_utils.subindex(s.values, sorted),
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
    """
    Align the DataFrame `df2` on the primary key column `pk_name` of
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
    """
    Compare two DataFrames element-wise for equality. Returns a DataFrame of
    Boolean values indicating whether the elements are equal.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first DataFrame to compare.
    df2 : pd.DataFrame
        The second DataFrame to compare.

    Returns
    -------
    pd.DataFrame
        A DataFrame of Boolean values indicating equality between elements.

    Raises
    ------
    ValueError
        If the DataFrames cannot be compared.

    Examples
    --------
    >>> df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> df2 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> df_eq(df1, df2)
        A	B
    0	True	True
    1	True	True
    2	True	True
    >>> df3 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 7]})
    >>> df_eq(df1, df3)
        A	B
    0	True	True
    1	True	True
    2	True	False
    """
    try:
        return (df1.isnull() & df1.isnull()) | (df1 == df2)
    except ValueError as e:
        print("DataFrames cannot be compared.")
        print(f"Caught ValueError: {e}")
        return pd.DataFrame([False])


def df_neq(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Compare two DataFrames element-wise for inequality. Returns a DataFrame of
    Boolean values indicating whether the elements are not equal.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first DataFrame to compare.
    df2 : pd.DataFrame
        The second DataFrame to compare.

    Returns
    -------
    pd.DataFrame
        A DataFrame of Boolean values indicating inequality between elements.

    Examples
    --------
    >>> df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> df2 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> df_neq(df1, df2)
        A	B
    0	False	False
    1	False	False
    2	False	False
    >>> df3 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 7]})
    >>> df_neq(df1, df3)
        A	B
    0	False	False
    1	False	False
    2	False	True
    """
    return ~df_eq(df1, df2) 


def check_dtypes_alignment(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """
    Check if the data types of columns in two DataFrames are aligned.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first DataFrame.
    df2 : pd.DataFrame
        The second DataFrame.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the DataFrames cannot be compared.

    Examples
    --------
    >>> df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> df2 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> check_dtypes_alignment(df1, df2)
    dtypes are aligned
    >>> df3 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 7.0]})
    >>> check_dtypes_alignment(df1, df3)
    dtypes diffs:
    B    int64
    dtype: object
    B    float64
    dtype: object
    """
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


def replace_chars(
    s1: pd.Series,
    s2: pd.Series,
    replacement: str = ""
) -> pd.Series:
    """
    Replace occurrences of characters from s1 with an empty string in s2.

    This function iterates over each pair of elements from s1 and s2 and replaces
    characters from s1 with an empty string in the corresponding elements of s2.
    If s1 and s2 have different lengths, the resulting Series will be the same
    length as the shorter of the two input Series.

    Parameters
    ----------
    s1 : pd.Series
        The first Series containing characters to be replaced in s2.
    s2 : pd.Series
        The second Series in which characters will be replaced.
    replacement : str, optional
        The character or string to use as a replacement for characters from s1.
        Defaults to an empty string.

    Returns
    -------
    pd.Series
        A new Series where characters from s1 have been replaced with the
        specified replacement.

    Examples
    --------
    >>> s1 = pd.Series(['a', 'b', 'c', 'd'])
    >>> s2 = pd.Series(['apple', 'banana', 'cherry', 'date'])
    >>> replace_chars(s1, s2)
    0     pple
    1    anana
    2    herry
    3      ate
    dtype: object

    >>> s3 = pd.Series(['b', 'f', 'g'])
    >>> s4 = pd.Series(['abc', 'def', 'ghi', 'jkl'])
    >>> replace_chars(s3, s4, replacement="-")
    0    a-c
    1    de-
    2    -hi
    dtype: object
    """
    s1 = s1.fillna(replacement)
    s2 = s2.fillna(replacement)
    return (pd.Series([
        x2.replace(x1, replacement)
        if x1 != x2 else ""
        for x1, x2 in zip(s1, s2)
    ], index=s1.index))



def micro_test_replace_char():
    """
    Perform a micro test of the `replace_chars` function.
    """
    s1 = pd.Series(["A", "B", "C", None, None])
    s2 = pd.Series(["A", "D", "C", "X", None])
    print(s2.where(s1 == s2, s2.str.cat(s1, sep="")))
    print(replace_chars(s1, s2))


def safe_diff_series(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """
    Compute the difference between two Pandas Series, handling different data types.

    This function subtracts the values in `s1` from the values in `s2`. If both Series
    contain numeric data types, a numerical subtraction is performed. If either of
    the Series contains non-numeric (e.g., string) data types, it uses the `replace_chars`
    function to remove the values in `s1` from `s2`.

    Parameters
    ----------
    s1 : pandas.Series
        The first input Series for subtraction.
    s2 : pandas.Series
        The second input Series for subtraction.

    Returns
    -------
    pandas.Series
        A Series containing the result of the subtraction or character removal.

    Examples
    --------
    >>> s1 = pd.Series([1, 2, 3, 4, 5])
    >>> s2 = pd.Series([5, 4, 3, 2, 1])
    >>> safe_diff_series(s1, s2)
    0    4
    1    2
    2    0
    3    -2
    4    -4
    dtype: int64

    >>> s1 = pd.Series(["A", "B", "C", None, None])
    >>> s2 = pd.Series(["A", "D", "C", "X", None])
    >>> safe_diff_series(s1, s2)
    0
    1    D
    2
    3    X
    4
    dtype: object
    """
    if (
        pd.api.types.is_numeric_dtype(s1)
        and pd.api.types.is_numeric_dtype(s2)
    ):
        return s2 - s1
    else:
        return replace_chars(s1, s2)


def safe_diff_dataframe(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the difference between two Pandas DataFrames, handling different data types.

    This function subtracts the values in `df1` from the values in `df2` column-wise.
    If the columns in `df1` and `df2` are of non-numeric data types (e.g., strings), it uses
    the `safe_diff_series` function to remove the values in `df1` from `df2` for each column.

    Parameters
    ----------
    df1 : pandas.DataFrame
        The first input DataFrame for subtraction.
    df2 : pandas.DataFrame
        The second input DataFrame for subtraction.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the result of the subtraction or character removal for each column.

    Examples
    --------
    >>> df1 = pd.DataFrame({"A": [1, 2, 3], "B": ["X", "Y", "Z"]})
    >>> df2 = pd.DataFrame({"A": [3, 2, 1], "B": ["Z", "Y", "X"]})
    >>> safe_diff_dataframe(df1, df2)
        A	B
    0	2	Z
    1	0	
    2	-2	X

    >>> df1 = pd.DataFrame({"A": ["A", "B", "C"], "B": ["X", "Y", "Z"]})
    >>> df2 = pd.DataFrame({"A": ["C", "B", "A"], "B": ["Z", "Y", "X"]})
    >>> safe_diff_dataframe(df1, df2)
        A	B
    0	C	Z
    1		
    2	A	X
    """
    return df1.apply(lambda s: safe_diff_series(s, df2[s.name]))
