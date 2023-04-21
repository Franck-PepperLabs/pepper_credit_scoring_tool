from typing import Optional, Union, Tuple

import pandas as pd
import numpy as np

from pandas.api.types import is_numeric_dtype
from pepper.feat_eng import nullify


""" Missing values
"""


def nullify_365243(
    data: Union[pd.Series, pd.DataFrame]
) -> None:
    """Replaces occurrences of the value 365243 by None in a pandas Series or
    DataFrame.
    
    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        The data to be modified inplace.
    
    Returns
    -------
    None
    """
    nullify(data, 365243)


def nullify_XNA(
    data: Union[pd.Series, pd.DataFrame]
) -> None:
    """Replaces occurrences of the value `'XNA'` by None in a pandas Series or
    DataFrame.
    
    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        The data to be modified inplace.
    
    Returns
    -------
    None
    """
    nullify(data, "XNA")


def negate_numerical_data(
    data: Union[pd.Series, pd.DataFrame]
) -> None:
    """Replaces all numerical values in a pandas DataFrame or Series by their
    opposite value.
    
    Parameters
    ----------
    data : pandas DataFrame or Series
        The DataFrame or Series to be modified.

    Returns
    -------
    None
    """
    if isinstance(data, pd.Series):
        if is_numeric_dtype(data):
            data.update(-data)
    elif isinstance(data, pd.DataFrame):
        # TODO : revoir pour un `update`
        # En l'état, produit un légitime `SettingWithCopyWarning`
        numeric_cols = data.select_dtypes(include="number").columns
        data[numeric_cols] = -data[numeric_cols]
    else:
        raise ValueError("Input must be a pandas Series or DataFrame.")


def drop_no_last_app_rows(prev_app: pd.DataFrame) -> None:
    """Drops rows from a DataFrame that do not represent the last application
    per contract.

    Parameters
    ----------
    prev_app : pd.DataFrame
        The DataFrame containing previous loan applications data.

    Returns
    -------
    None
        The function does not return anything, but rather modifies the input
        DataFrame in place.

    Note:
        This function drops rows from the input DataFrame based on two
        conditions:
        1. The value of the 'FLAG_LAST_APPL_PER_CONTRACT' column is 'N'
        2. The value of the 'NFLAG_LAST_APPL_IN_DAY' column is 0
        Rows that meet either of these conditions are dropped from the
        DataFrame.
    """
    bad_1 = prev_app.FLAG_LAST_APPL_PER_CONTRACT
    bad_2 = prev_app.NFLAG_LAST_APPL_IN_DAY
    mask = (bad_1 == "N") | (bad_2 == 0)
    drop_index = prev_app[mask].index
    prev_app.drop(index=drop_index, inplace=True)


""" `bureau_balance`
"""


def divide_rle(
    rle: Optional[np.ndarray]
) -> Tuple[int, int, int, Optional[Tuple[Tuple[str, int]]]]:
    """Given a Run Length Encoded (RLE) numpy array, returns a tuple containing
    the number of closed frames, the total number of tracked frames, the number
    of frames not tracked, and a tuple representation of the RLE array
    containing the track status and frame counts for each frame.

    Parameters
    ----------
    rle : np.ndarray
        The RLE array to be divided.

    Returns
    -------
    Tuple[int, int, int, Optional[Tuple[Tuple[str, int]]]]
        A tuple containing the number of closed frames, the total number of
        tracked frames, the number of frames not tracked, and a tuple
        representation of the RLE array containing the track status and frame
        counts for each frame.
    
    See also
    --------
    `pepper.feat_eng.row_rle`
    """
    n_closed = n_tracked = n_notrack = 0
    profile = None
    if rle is not None and rle.shape[0] > 0:
        start = 0
        end = rle.shape[0]

        # Check if the RLE array starts with a closed frame and if so,
        # add it to the count of closed frames
        if rle[0][0] in ["C", "☗"] or pd.isnull(rle[0][0]):
            n_closed = rle[0][1]
            start = 1

        # Check if the RLE array ends with an empty frame and if so,
        # add it to the count of untracked frames
        if rle[-1][0] in ["C", "☗"] or pd.isnull(rle[-1][0]):
            n_notrack = rle[-1][1]
            end = -1
        
        # Extract the profile by slicing the RLE array,
        # and compute the total number of tracked frames
        profile = rle[start:end]
        n_tracked = np.sum(profile[:, 1])

        # Convert the profile to a tuple of tuples for more elegant formatting
        profile = tuple(tuple(x) for x in profile)
    
    # Return the results in a tuple
    return n_closed, n_tracked, n_notrack, profile
