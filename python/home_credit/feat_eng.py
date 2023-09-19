from typing import Optional, Union, Tuple, List

import pandas as pd
import numpy as np

from pandas.api.types import is_numeric_dtype

from pepper.feat_eng import nullify

from home_credit.load import get_table
from home_credit.utils import _not_in_subs_table_idx


""" Missing values
"""


def nullify_365243(
    data: Union[pd.Series, pd.DataFrame]
) -> None:
    """
    Replace occurrences of the value 365243 by None in a pandas Series or
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
    """
    Replaces occurrences of the value `'XNA'` by None in a pandas Series or
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
    """
    Replace all numerical values in a pandas DataFrame or Series by their
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
    """
    Drop rows from a DataFrame that do not represent the last application
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


""" `*_balance`
"""


def eval_contracts_status(x: str) -> str:
    """
    Evaluate the status of credit contracts in the context of post-processing NAME_CONTRACT_STATUS.

    This function is designed for post-processing the 'NAME_CONTRACT_STATUS' column to reduce
    combinations resulting from concatenation of contract statuses. The function evaluates
    the status of credit contracts and returns one of the following: 'Active', 'Signed',
    'Completed', 'Demand', or the original value if none of these statuses are found.

    Parameters
    ----------
    x : str
        The input value representing the status of credit contracts in a concatenated form.

    Returns
    -------
    str
        The evaluated status of the credit contracts.

    Notes
    -----
    - If the input value `x` is `NaN` or an integer, the function returns `NaN`.
    - The function searches for specific keywords ('Active', 'Signed', 'Completed', 'Demand') in
      the input string and returns the first match found.
    - If none of the keywords are found, the original input value `x` is returned.

    Examples
    --------
    >>> eval_contracts_status("Active and Completed")
    'Active'
    
    >>> eval_contracts_status("Signed and Demand")
    'Signed'

    >>> eval_contracts_status("Unknown")
    'Unknown'

    >>> eval_contracts_status(np.nan)
    NaN

    Context
    -------
    This function is intended for post-processing the 'NAME_CONTRACT_STATUS' column when dealing
    with aggregated data. It helps reduce the complexity of contract status combinations.

    Usage
    -----
    Example usage within the context of aggregating 'multis' data:
    ```
    from home_credit.feat_eng import eval_contracts_status

    grouped_multis.NAME_CONTRACT_STATUS = (
        grouped_multis.NAME_CONTRACT_STATUS
        .apply(eval_contracts_status)
    )
    ```
    """
    if pd.isnull(x) or isinstance(x, int):
        return np.nan
    return next(
        (
            prior
            for prior in ["Active", "Signed", "Completed", "Demand"]
            if prior in x
        ),
        x,
    )


def divide_rle(
    rle: Optional[np.ndarray]
) -> Tuple[int, int, int, Optional[Tuple[Tuple[str, int]]]]:
    """
    Given a Run Length Encoded (RLE) numpy array, returns a tuple containing
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


""" Derived features
"""


def _is_in_subs_table(
    main_table_name: str,
    subs_table_name: str,
    key: str
) -> pd.Series:
    """
    Determine whether a record from the main table is present in a
    subsidiary table.

    Parameters
    ----------
    main_table_name : str
        Name of the main table.
    subs_table_name : str
        Name of the subsidiary table.
    key : str
        Name of the primary key column.

    Returns
    -------
    pd.Series
        A boolean series indicating whether a record is present in the
        subsidiary table or not.
    """
    index = _not_in_subs_table_idx(main_table_name, subs_table_name, key)
    return ~get_table(main_table_name)[key].isin(index)


def is_in_ext(
    key: str,
    main_table_name: str,
    subs_table_names: List[str],
    var_names: Union[List[str], None] = None
) -> pd.DataFrame:
    """
    Return a DataFrame containing new features indicating if a record is
    present in each subsidiary table.

    Parameters
    ----------
    key : str
        Name of the primary key column shared by the main table and subsidiary
        tables.
    main_table_name : str
        Name of the main table.
    subs_table_names : List[str]
        List of names of the subsidiary tables to check against.
    var_names : Union[List[str], None], optional
        List of names for the new columns indicating whether a record is
        present in each table, by default None. If None, generate column names
        automatically.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing new features indicating if a record is present
        in each subsidiary table.
    Examples
    --------
    >>> subs_table_names = [
    >>>     "bureau", "previous_application", "pos_cash_balance",
    >>>     "credit_card_balance", "installments_payments"
    >>> ]
    >>> app_is_in = is_in_ext("SK_ID_CURR", "application", subs_table_names)

    """
    """subs_table_names = [
        "bureau", "previous_application", "pos_cash_balance",
        "credit_card_balance", "installments_payments"
    ]
    var_names = ["in_B", "in_PA", "in_PCB", "in_CCB", "in_IP"]"""
    if var_names is None:
        var_names = [
            f"in_{''.join([a[0].upper()for a in name.split('_')])}"
            for name in subs_table_names
        ]
    return pd.concat(
        [
            _is_in_subs_table(main_table_name, subs_table_name, key)
            .rename(var_name)
            for subs_table_name, var_name in zip(subs_table_names, var_names)
        ],
        axis=1
    )
