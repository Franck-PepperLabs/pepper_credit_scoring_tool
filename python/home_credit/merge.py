from typing import *

import pandas as pd
import numpy as np

from pepper.utils import display_key_val

from home_credit.load import get_table, get_target
from home_credit.feat_eng import eval_contracts_status


def targetize(sk_id_curr: pd.Series) -> pd.Series:
    """Map SK_ID_CURR values to their corresponding TARGET values in the main
    table.

    Parameters
    ----------
    sk_id_curr : pd.Series
        Series of `SK_ID_CURR` values.

    Returns
    -------
    pd.Series
        Series of corresponding `TARGET` values.
    """
    curr_tgt_map = get_target().TARGET
    return sk_id_curr.map(curr_tgt_map)


# La table `bureau_balance` a besoin d'être currentized
# avant de pouvoir être targetized !
def currentize(sk_id_bur: pd.Series) -> pd.Series:
    bur_curr_map = get_table("bureau").copy()[["SK_ID_BUREAU", "SK_ID_CURR"]]
    bur_curr_map.set_index("SK_ID_BUREAU", inplace=True)
    bur_curr_map = bur_curr_map.SK_ID_CURR
    return sk_id_bur.map(bur_curr_map)




def get_singl_and_multi_index(
    table_name: str,
    prev_sk: str,
    curr_sk: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns four dataframes containing unique SK_ID_PREV and SK_ID_CURR,
    split between those that appear only once in the original data, and those
    that appear multiple times.

    Parameters
    ----------
    table_name : str
        Name of the table to load from the home_credit package.
    prev_sk : str
        Name of the column containing the SK_ID_PREV.
    curr_sk : str
        Name of the column containing the SK_ID_CURR.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple of four dataframes:
            - singl_prev_idx : Unique SK_ID_PREV that appear only once in the
                original data.
            - multi_prev_idx : SK_ID_PREV that appear multiple times in the
                original data.
            - singl_curr_idx : Unique SK_ID_CURR that appear only once in the
                original data.
            - multi_curr_idx : SK_ID_CURR that appear multiple times in the
                original data.

    """
    sk_couples = get_table(table_name)[[prev_sk, curr_sk]].copy()
    sk_index = sk_couples.drop_duplicates()
    dupl_prev_idx = sk_index.duplicated(subset=prev_sk, keep=False)
    dupl_curr_idx = sk_index.duplicated(subset=curr_sk, keep=False)
    singl_prev_idx = sk_index[~dupl_curr_idx]
    multi_prev_idx = sk_index[dupl_curr_idx]
    singl_curr_idx = sk_index[~dupl_prev_idx]
    multi_curr_idx = sk_index[dupl_prev_idx]
    return singl_prev_idx, multi_prev_idx, singl_curr_idx, multi_curr_idx


def curr_prev_unicity_report(
    singl_prev_idx: pd.DataFrame,
    multi_prev_idx: pd.DataFrame,
    singl_curr_idx: pd.DataFrame,
    multi_curr_idx: pd.DataFrame
) -> None:
    """Displays a report of the unicity of SK_ID_CURR and SK_ID_PREV in the
    data.

    Parameters
    ----------
    singl_prev_idx : pd.DataFrame
        Unique SK_ID_PREV that appear only once in the original data.
    multi_prev_idx : pd.DataFrame
        SK_ID_PREV that appear multiple times in the original data.
    singl_curr_idx : pd.DataFrame
        Unique SK_ID_CURR that appear only once in the original data.
    multi_curr_idx : pd.DataFrame
        SK_ID_CURR that appear multiple times in the original data.

    Returns
    -------
    None

    """
    n_singl_prev = singl_prev_idx.shape[0]
    n_multi_prev = multi_prev_idx.shape[0]
    n_singl_curr = singl_curr_idx.shape[0]
    n_multi_curr = multi_curr_idx.shape[0]
    n = n_singl_prev + n_multi_prev
    display_key_val("number of unique (curr, prev)", n)
    display_key_val("number of curr with more than 1 prev", n_multi_prev)    
    display_key_val("number of curr with one prev", n_singl_prev)
    display_key_val("number of curr with more than 1 prev (in %)", round(100 * n_multi_prev / n, 1))
    display_key_val("number of prev with more than 1 curr", n_multi_curr)
    display_key_val("number of prev with one curr", n_singl_curr)
    display_key_val("number of prev with more than 1 curr (in %)", round(100 * n_multi_curr / n, 1))


def ip_months_balance_builder(data: pd.DataFrame) -> pd.Series:
    """Builds the `MONTHS_BALANCE` column based on the provided data.

    Parameters
    ----------
    data : pd.DataFrame
        The data to build the `MONTHS_BALANCE` column from.

    Returns
    -------
    pd.Series
        The `MONTHS_BALANCE` column.
    """
    gregorian_month = 365.2425 / 12
    return -(data.DAYS_INSTALMENT // gregorian_month).astype(int)


def groupby_curr_months(
    table_name: str,
    months_balance_builder: callable = None,
    agg_dict: dict = None,
    include_singles: bool = False
) -> pd.DataFrame:
    """Groups rows by a combination of `SK_ID_CURR` and `MONTHS_BALANCE` based
    on the provided table, which is assumed to have a `SK_ID_PREV` column.

    Parameters
    ----------
    table_name : str
        The name of the table to group by.
    months_balance_builder : callable, optional
        A callable used to build the `MONTHS_BALANCE` column based on the
        provided data. Defaults to None, which means no `MONTHS_BALANCE` column
        will be added.
    agg_dict : dict, optional
        A dictionary of columns and corresponding aggregation functions used to
        aggregate the `n_PREV` column and any additional columns specified in
        the `agg_dict`. Defaults to None, which means only the `n_PREV` column
        will be aggregated.
    include_singles : bool, optional
        If False, aggregate only multi-PREV rows.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the grouped data.
    
    Example
    -------
    >>> def ip_months_balance_builder(data):
    >>>     gregorian_month = 365.2425 / 12
    >>>     return -(data.DAYS_INSTALMENT // gregorian_month).astype(int)
    >>>
    >>> grouped_data = groupby_curr_months(
    >>>     "installments_payments",
    >>>     ip_months_balance_builder,
    >>>     {
    >>>         "NUM_INSTALMENT_VERSION": "max",
    >>>         "NUM_INSTALMENT_NUMBER": "max",
    >>>         "DAYS_INSTALMENT": "median",
    >>>         "DAYS_ENTRY_PAYMENT": "median",
    >>>         "AMT_INSTALMENT": "sum",
    >>>         "AMT_PAYMENT": "sum"
    >>>     }
    >>> )
    """
    # Load data
    data = get_table(table_name).copy()

    # Insert the aggregation counter
    data.insert(0, "n_PREV", 1)

    # Calculate or adjust MONTHS_BALANCE
    if months_balance_builder is None:
        data.MONTHS_BALANCE = -data.MONTHS_BALANCE
    else:
        data.insert(0, "MONTHS_BALANCE", months_balance_builder(data))

    if include_singles:  # No distinction
        data_multis = data
    else:  # Protect singles
        # Split into subtables : single-PREV vs multi-PREV
        singl_prev_idx, _, _, _ = get_singl_and_multi_index(
            table_name, "SK_ID_PREV", "SK_ID_CURR"
        )
        is_single_mask = data.SK_ID_CURR.isin(singl_prev_idx.SK_ID_CURR)
        data_multis = data[~is_single_mask]
        data_singles = data[is_single_mask]

    # Group multi-PREV by CURR-MONTH
    gpby = (
        data_multis.drop(columns="SK_ID_PREV")
        .groupby(by=["SK_ID_CURR", "MONTHS_BALANCE"])
    )

    # Aggregate data
    if agg_dict is None:
        grouped_multis = gpby.sum()
    if agg_dict is not None:
        _agg_dict = {"n_PREV": "sum"}
        _agg_dict.update(agg_dict)
        grouped_multis = gpby.agg(_agg_dict)

    # Post-process `NAME_CONTRACT_STATUS` (if exists)
    if "NAME_CONTRACT_STATUS" in grouped_multis.columns:
        grouped_multis.NAME_CONTRACT_STATUS = (
            grouped_multis.NAME_CONTRACT_STATUS
            .apply(eval_contracts_status)
        )

    grouped_multis.reset_index(level=1, inplace=True)
    if include_singles:
        return grouped_multis
    else:  # Align subtables
        grouped_singles = data_singles.drop(columns="SK_ID_PREV")
        # Trick to reposition MONTHS_BALANCE in the right position (first column)
        grouped_singles.set_index(
            ["MONTHS_BALANCE", "SK_ID_CURR"],
            drop=True, inplace=True
        )
        grouped_singles.reset_index(level=0, inplace=True)

        # Stack them and return
        grouped_data = pd.concat([grouped_multis, grouped_singles])
        grouped_data.sort_index(inplace=True)
        return grouped_data


def groupby_curr_num(
    table_name: str,
    months_balance_builder: callable = None,
    agg_dict: dict = None,
    include_singles: bool = False
) -> pd.DataFrame:
 
    # Load data
    data = get_table(table_name).copy()

    # Insert the aggregation counter
    data.insert(0, "n_PREV", 1)

    # Calculate or adjust MONTHS_BALANCE
    if months_balance_builder is None:
        data.MONTHS_BALANCE = -data.MONTHS_BALANCE
    else:
        data.insert(0, "MONTHS_BALANCE", months_balance_builder(data))

    if include_singles:  # No distinction
        data_multis = data
    else:  # Protect singles
        # Split into subtables : single-PREV vs multi-PREV
        singl_prev_idx, _, _, _ = get_singl_and_multi_index(
            table_name, "SK_ID_PREV", "SK_ID_CURR"
        )
        is_single_mask = data.SK_ID_CURR.isin(singl_prev_idx.SK_ID_CURR)
        data_multis = data[~is_single_mask]
        data_singles = data[is_single_mask]

    # Group multi-PREV by CURR-MONTH
    gpby = (
        data_multis.drop(columns="SK_ID_PREV")
        .groupby(by=["SK_ID_CURR", "NUM_INSTALMENT_NUMBER"])
    )

    # Aggregate data
    if agg_dict is None:
        grouped_multis = gpby.sum()
    if agg_dict is not None:
        _agg_dict = {"n_PREV": "sum"}
        _agg_dict.update(agg_dict)
        grouped_multis = gpby.agg(_agg_dict)

    # Post-process `NAME_CONTRACT_STATUS` (if exists)
    if "NAME_CONTRACT_STATUS" in grouped_multis.columns:
        grouped_multis.NAME_CONTRACT_STATUS = (
            grouped_multis.NAME_CONTRACT_STATUS
            .apply(eval_contracts_status)
        )

    grouped_multis.reset_index(level=1, inplace=True)
    if include_singles:
        return grouped_multis
    else:  # Align subtables
        grouped_singles = data_singles.drop(columns="SK_ID_PREV")
        # Trick to reposition NUM_INSTALMENT_VERSION in the right position (first column)
        grouped_singles.set_index(
            ["NUM_INSTALMENT_NUMBER", "SK_ID_CURR"],
            drop=True, inplace=True
        )
        grouped_singles.reset_index(level=0, inplace=True)

        # Stack them and return
        grouped_data = pd.concat([grouped_multis, grouped_singles])
        grouped_data.sort_index(inplace=True)
        return grouped_data
    