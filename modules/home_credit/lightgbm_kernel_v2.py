
import time
from contextlib import contextmanager
import lightgbm as lgb

import numpy as np
import pandas as pd


@contextmanager
def timer(title):
    t_0 = time.time()
    yield
    print(f"{title} - done in {time.time() - t_0:.0f}s")


def get_categorical_vars(
    df: pd.DataFrame,
    dtype: type = object,
    max_modalities: int = None
) -> list:
    """Get a list of categorical variables in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to search for categorical variables.
    dtype : type, optional
        The data type(s) to search for. If None, all columns will be searched.
        If a single type is given, only columns of that type will be returned.
        If a list or tuple of types is given, columns of any of those types
        will be returned. Defaults to object.
    max_modalities : int, optional
        The maximum number of unique values a column can have to be considered
        categorical. If None, all columns that match the dtype criteria will
        be returned. Defaults to None.

    Returns
    -------
    list
        A list of column names that meet the criteria for categorical variables.

    """
    if dtype is None and max_modalities is None:
        # Return all columns if no dtype or max_modalities is specified
        return list(df.columns)
    elif dtype is None:
        # Return columns with unique values less than or equal to max_modalities
        return [col for col in df.columns if df[col].nunique() <= max_modalities]
    elif max_modalities is None:
        if isinstance(dtype, (list, tuple, np.ndarray, pd.Index, pd.Series)):
            # Return columns with data types that match any of the given types
            return [col for col in df.columns if df[col].dtype in dtype]
        else:
            # Return columns with the specified data type
            return [col for col in df.columns if df[col].dtype == dtype]
    else:
        if isinstance(dtype, (list, tuple, np.ndarray, pd.Index, pd.Series)):
            # Return columns with data types that match any of the given types and
            # have unique values less than or equal to max_modalities
            return [
                col for col in df.columns
                if (
                    df[col].dtype in dtype
                    and df[col].nunique() <= max_modalities
                )
            ]
        else:
            # Return columns with the specified data type and
            # have unique values less than or equal to max_modalities
            return [
                col for col in df.columns
                if (
                    df[col].dtype == dtype
                    and df[col].nunique() <= max_modalities
                )
            ]


def one_hot_encode_all_cats(
    df: pd.DataFrame,
    columns: list[str] = None,
    dummy_na: bool = True,
    drop_first: bool = True,
    dtype: type = np.int8,
    sparse: bool = True,
    discard_constants: bool = False
) -> pd.DataFrame:
    """One-hot encode all categorical columns in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame)
        The DataFrame to be encoded.
    columns : list[str], optional
        The list of column names to be encoded.
        If None, all categorical columns will be encoded. Defaults to None.
    dummy_na : bool, optional
        Add a column to indicate NaNs. Defaults to True.
    drop_first : bool, optional
        Whether to get k-1 dummies out of k categorical levels by removing the
        first level. Defaults to True.
    dtype : type, optional
        The data type of the encoded columns. Defaults to `np.int8`.
    sparse : bool, optional
        Whether to return a sparse matrix or not. Defaults to True.
    discard_constants : bool, optional
        Whether to remove the potentially generated constant columns
        by `dummy_na`. Defaults to True.

    Returns
    -------
    pd.DataFrame
        The one-hot encoded DataFrame.

    """
    ohe_df = pd.get_dummies(
        df, columns=columns, dummy_na=dummy_na,
        drop_first=drop_first, dtype=dtype, sparse=sparse
    )
    # Removing the potentially generated constant columns by `dummy_na`
    if discard_constants:
        const_cols = ohe_df.apply(pd.Series.nunique)
        const_cols = const_cols[const_cols == 1]
        ohe_df.drop(columns=const_cols.index, inplace=True)
    return ohe_df
