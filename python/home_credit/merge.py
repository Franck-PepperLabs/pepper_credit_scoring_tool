"""Module: home_credit.merge

This module provides a set of functions for merging and aggregating Home Credit data tables.
It enables the creation of consolidated datasets based on different criteria, including SK_ID_CURR,
MONTHS_BALANCE, and NUM_INSTALMENT_NUMBER. The module offers flexibility in defining aggregation
methods and supports the inclusion or exclusion of unique rows in the aggregated data.

Functions:
- `targetize(sk_id_curr: pd.Series) -> pd.Series`:
    Map `SK_ID_CURR` values to their corresponding `TARGET` values in the main table.

- `currentize(sk_id_bur: pd.Series) -> pd.Series`:
    Map `SK_ID_BUREAU` values to their corresponding `SK_ID_CURR` values,
    currentizing the bureau table.

- `get_unique_and_multi_index(table_name: str, prev_sk: str, curr_sk: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]`:
    Return four dataframes containing unique `SK_ID_PREV` and `SK_ID_CURR`
    combinations, split between those that appear only once and those that
    appear multiple times in the original data.

- `curr_prev_uniqueness_report(unique_prev_idx: pd.DataFrame, multi_prev_idx: pd.DataFrame, unique_curr_idx: pd.DataFrame, multi_curr_idx: pd.DataFrame) -> None`:
    Display a report on the uniqueness of `SK_ID_CURR` and `SK_ID_PREV` in the data.

- `ip_months_balance_builder(data: pd.DataFrame) -> pd.Series`:
    Build the `MONTHS_BALANCE` column based on the provided data.

- `_combine_grouped_data(data_uniques, grouped_multis, pivot_col)`:
    Combine grouped dataframes of unique and multi-PREV rows based on a specified pivot column.

- `_groupby_curr_pivot(table_name: str, pivot_col: str, months_balance_builder: callable = None, agg_dict: dict = None, include_uniques: bool = False) -> pd.DataFrame`:
    Group rows by a combination of `SK_ID_CURR` and the specified pivot column
    and aggregate data based on the provided table.

- `groupby_curr_months(table_name: str, months_balance_builder: callable = None, agg_dict: dict = None, include_uniques: bool = False) -> pd.DataFrame`:
    Group rows by a combination of `SK_ID_CURR` and `MONTHS_BALANCE`
    and aggregate data based on the provided table.

- `groupby_curr_num(table_name: str, months_balance_builder: callable = None, agg_dict: dict = None, include_uniques: bool = False) -> pd.DataFrame`:
    Group rows by a combination of `SK_ID_CURR` and `NUM_INSTALMENT_NUMBER`
    and aggregate data based on the provided table.

This module simplifies the process of merging and aggregating Home Credit data
for analysis, providing flexibility and options for creating consolidated
datasets for various analytical needs.
"""

from typing import Tuple

import pandas as pd
import numpy as np

from pepper.utils import display_key_val

from home_credit.load import get_table, get_target
from home_credit.feat_eng import eval_contracts_status


def map_bur_to_curr(sk_id_bur: pd.Series) -> pd.Series:
    """
    DEPRECATED Use home_credit.tables.map_bur_to_curr instead
    
    Map `SK_ID_BUR` values to their corresponding `SK_ID_CURR` values
    by using a mapping table extracted from the 'bureau' table.

    Note - The 'bureau_balance' table needs to be currentized before it
    can be targetized.

    Parameters
    ----------
    sk_id_bur : pd.Series
        Series of `SK_ID_BUR` values.

    Returns
    -------
    pd.Series
        Series of corresponding `SK_ID_CURR` values.
    """
    bur_curr_map = get_table("bureau").copy()[["SK_ID_BUREAU", "SK_ID_CURR"]]
    bur_curr_map.set_index("SK_ID_BUREAU", inplace=True)
    bur_curr_map = bur_curr_map.SK_ID_CURR.astype(np.uint32)
    return sk_id_bur.map(bur_curr_map)


def currentize(data: pd.DataFrame) -> None:
    """
    DEPRECATED Use home_credit.tables.currentize instead
    
    Currentize a DataFrame by adding the 'SK_ID_CURR' column based on the
    'SK_ID_BUREAU' values.

    This function adds a new column 'SK_ID_CURR' to the DataFrame by mapping
    'SK_ID_BUREAU' values to their corresponding 'SK_ID_CURR' values using
    a mapping table extracted from the 'bureau' table.

    Note: The 'bureau_balance' table needs to be currentized before it can
    be targetized.

    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame to currentize, which should contain the 'SK_ID_BUREAU' column.

    Raises:
    -------
    ValueError
        If the 'SK_ID_BUREAU' column is not found in the DataFrame.

    Returns:
    --------
    None
    """
    # Check if data contains the 'SK_ID_BUREAU' column, otherwise raise an exception
    if "SK_ID_BUREAU" not in data.columns:
        raise ValueError("The DataFrame does not contain the 'SK_ID_BUREAU' column.")
    
    data.insert(0, "SK_ID_CURR", map_bur_to_curr(data.SK_ID_BUREAU))


def map_curr_to_target(sk_id_curr: pd.Series) -> pd.Series:
    """
    DEPRECATED Use home_credit.tables.map_curr_to_target instead
    
    Map `SK_ID_CURR` values to their corresponding `TARGET` values in the main
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
    curr_tgt_map = get_target().TARGET.astype(np.int8)
    return sk_id_curr.map(curr_tgt_map)


def targetize(data: pd.DataFrame) -> None:
    """
    DEPRECATED Use home_credit.tables.targetize instead
    
    Targetize a DataFrame by adding the 'TARGET' column based on the
    'SK_ID_CURR' values.

    This function adds a new column 'TARGET' to the DataFrame by mapping
    'SK_ID_CURR' values to their corresponding 'TARGET' values in the main table.

    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame to targetize, which should contain the 'SK_ID_CURR' column.

    Raises:
    -------
    ValueError
        If the 'SK_ID_CURR' column is not found in the DataFrame.

    Returns:
    --------
    None
    """
    # Check if data contains the 'SK_ID_CURR' column, otherwise raise an exception
    if "SK_ID_CURR" not in data.columns:
        raise ValueError("The DataFrame does not contain the 'SK_ID_CURR' column.")
    
    data.insert(0, "TARGET", map_curr_to_target(data.SK_ID_CURR))


# TODO : ajoutée pour le cas de bureau_balance : mieux intégrer
def _get_unique_and_multi_index(
    table: pd.DataFrame,
    subs_sk: str,
    main_sk: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sk_couples = table[[subs_sk, main_sk]].copy()
    sk_index = sk_couples.drop_duplicates()
    dupl_subs_idx = sk_index.duplicated(subset=subs_sk, keep=False)
    dupl_main_idx = sk_index.duplicated(subset=main_sk, keep=False)
    unique_prev_idx = sk_index[~dupl_main_idx]
    multi_prev_idx = sk_index[dupl_main_idx]
    unique_curr_idx = sk_index[~dupl_subs_idx]
    multi_curr_idx = sk_index[dupl_subs_idx]
    return unique_prev_idx, multi_prev_idx, unique_curr_idx, multi_curr_idx


def get_unique_and_multi_index(
    table_name: str,
    prev_sk: str,
    curr_sk: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return four dataframes containing unique `SK_ID_PREV` and `SK_ID_CURR`,
    split between those that appear only once in the original data,
    and those that appear multiple times.

    Parameters
    ----------
    table_name : str
        The name of the table to load from the 'home_credit' package.
    prev_sk : str
        The name of the column containing the previous SK_ID (SK_ID_PREV).
    curr_sk : str
        The name of the column containing the current SK_ID (SK_ID_CURR).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple of four dataframes:
            - unique_prev_idx : Unique previous SK_ID (SK_ID_PREV) that appear
              only once in the original data.
            - multi_prev_idx : Previous SK_ID (SK_ID_PREV) that appear multiple
              times in the original data.
            - unique_curr_idx : Unique current SK_ID (SK_ID_CURR) that appear
              only once in the original data.
            - multi_curr_idx : Current SK_ID (SK_ID_CURR) that appear multiple
              times in the original data.
    """
    """
    sk_couples = get_table(table_name)[[prev_sk, curr_sk]].copy()
    sk_index = sk_couples.drop_duplicates()
    dupl_prev_idx = sk_index.duplicated(subset=prev_sk, keep=False)
    dupl_curr_idx = sk_index.duplicated(subset=curr_sk, keep=False)
    unique_prev_idx = sk_index[~dupl_curr_idx]
    multi_prev_idx = sk_index[dupl_curr_idx]
    unique_curr_idx = sk_index[~dupl_prev_idx]
    multi_curr_idx = sk_index[dupl_prev_idx]
    return unique_prev_idx, multi_prev_idx, unique_curr_idx, multi_curr_idx
    """
    return _get_unique_and_multi_index(get_table(table_name), prev_sk, curr_sk)


def curr_prev_uniqueness_report(
    unique_prev_idx: pd.DataFrame,
    multi_prev_idx: pd.DataFrame,
    unique_curr_idx: pd.DataFrame,
    multi_curr_idx: pd.DataFrame
) -> None:
    """
    Display a report of the uniqueness of SK_ID_CURR and SK_ID_PREV in the data.

    Parameters
    ----------
    unique_prev_idx : pd.DataFrame
        Unique SK_ID_PREV that appear only once in the original data.
    multi_prev_idx : pd.DataFrame
        SK_ID_PREV that appear multiple times in the original data.
    unique_curr_idx : pd.DataFrame
        Unique SK_ID_CURR that appear only once in the original data.
    multi_curr_idx : pd.DataFrame
        SK_ID_CURR that appear multiple times in the original data.

    Returns
    -------
    None

    """
    n_unique_prev = unique_prev_idx.shape[0]
    n_multi_prev = multi_prev_idx.shape[0]
    n_unique_curr = unique_curr_idx.shape[0]
    n_multi_curr = multi_curr_idx.shape[0]
    n = n_unique_prev + n_multi_prev
    display_key_val(
        "number of unique (curr, prev)              ",
        n
    )
    display_key_val(
        "number of curr with more than 1 prev       ",
        n_multi_prev
    )    
    display_key_val(
        "number of curr with one prev               ",
        n_unique_prev
    )
    display_key_val(
        "number of curr with more than 1 prev (in %)",
        round(100 * n_multi_prev / n, 1)
    )
    display_key_val(
        "number of prev with more than 1 curr       ",
        n_multi_curr
    )
    display_key_val(
        "number of prev with one curr               ",
        n_unique_curr
    )
    display_key_val(
        "number of prev with more than 1 curr (in %)",
        round(100 * n_multi_curr / n, 1)
    )


def ip_months_balance_builder(data: pd.DataFrame) -> pd.Series:
    """
    Build the `MONTHS_BALANCE` column based on the provided data.

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


def _combine_grouped_data(
    data_uniques: pd.DataFrame,
    grouped_multis: pd.DataFrame,
    pivot_col: str
) -> pd.DataFrame:
    """
    Combine grouped data from unique and multi-PREV rows based on the pivot column.

    Parameters
    ----------
    data_uniques : pd.DataFrame
        DataFrame containing data from unique SK_ID_PREV rows.
    grouped_multis : pd.DataFrame
        DataFrame containing data from multi-PREV rows.
    pivot_col : str
        The name of the pivot column used for grouping.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with the data from unique and multi-PREV rows.

    Notes
    -----
    This function is used to combine grouped data from unique and multi-PREV rows
    based on the specified pivot column. It takes data from 'data_uniques' and
    'grouped_multis', stacks them together, and sorts the index.

    """
    # Extract data from unique rows and prepare for stacking
    grouped_uniques = data_uniques.drop(columns="SK_ID_PREV")
    # Trick to reposition pivot_col in the right position (first column)
    grouped_uniques.set_index([pivot_col, "SK_ID_CURR"], drop=True, inplace=True)
    grouped_uniques.reset_index(level=0, inplace=True)

    # Stack unique and multi-PREV data together and sort the index
    grouped_data = pd.concat([grouped_multis, grouped_uniques])
    grouped_data.sort_index(inplace=True)

    return grouped_data


def _groupby_curr_pivot(
    table_name: str,
    pivot_col: str,
    months_balance_builder: callable = None,
    agg_dict: dict = None,
    include_uniques: bool = False
) -> pd.DataFrame:
    """
    Group rows by a combination of `SK_ID_CURR` and 'pivot_col' based on the provided table.

    Parameters
    ----------
    table_name : str
        The name of the table to group by.
    pivot_col : str
        The name of the pivot column used for grouping.
    months_balance_builder : callable, optional
        A callable used to build the `MONTHS_BALANCE` column based on the provided data.
        Defaults to None, which means no `MONTHS_BALANCE` column will be added.
    agg_dict : dict, optional
        A dictionary of columns and corresponding aggregation functions used to aggregate
        the `n_PREV` column and any additional columns specified in the `agg_dict`.
        Defaults to None, which means only the `n_PREV` column will be aggregated.
    include_uniques : bool, optional
        If False, aggregate only multi-PREV rows.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the grouped data.

    Notes
    -----
    This function groups rows by a combination of `SK_ID_CURR` and 'pivot_col' based on
    the provided table. It can optionally add a `MONTHS_BALANCE` column using the
    provided `months_balance_builder` function. It also supports aggregation of columns
    using the provided `agg_dict`. You can choose to include or exclude unique PREV rows
    based on the `include_uniques` parameter.

    Example
    -------
    >>> def ip_months_balance_builder(data):
    >>>     gregorian_month = 365.2425 / 12
    >>>     return -(data.DAYS_INSTALMENT // gregorian_month).astype(int)
    >>>
    >>> grouped_data = groupby_curr_pivot(
    >>>     "installments_payments",
    >>>     "MONTHS_BALANCE",
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

    if include_uniques:  # No distinction
        data_multis = data
    else:  # Protect uniques
        # Split into sub-tables : single-PREV vs multi-PREV
        unique_prev_idx, _, _, _ = get_unique_and_multi_index(
            table_name, "SK_ID_PREV", "SK_ID_CURR"
        )
        is_single_mask = data.SK_ID_CURR.isin(unique_prev_idx.SK_ID_CURR)
        data_multis = data[~is_single_mask]
        data_uniques = data[is_single_mask]

    # Group multi-PREV by pivot_col
    grouped = (
        data_multis.drop(columns="SK_ID_PREV")
        .groupby(by=["SK_ID_CURR", pivot_col])
    )

    # Aggregate data
    if agg_dict is None:
        grouped_multis = grouped.sum()
    if agg_dict is not None:
        _agg_dict = {"n_PREV": "sum"} | agg_dict
        grouped_multis = grouped.agg(_agg_dict)

    # Post-process `NAME_CONTRACT_STATUS` (if exists)
    if "NAME_CONTRACT_STATUS" in grouped_multis.columns:
        grouped_multis.NAME_CONTRACT_STATUS = (
            grouped_multis.NAME_CONTRACT_STATUS
            .apply(eval_contracts_status)
        )

    grouped_multis.reset_index(level=1, inplace=True)
    if include_uniques:
        return grouped_multis
    return _combine_grouped_data(data_uniques, grouped_multis, pivot_col)


def groupby_curr_months(
    table_name: str,
    months_balance_builder: callable = None,
    agg_dict: dict = None,
    include_uniques: bool = False
) -> pd.DataFrame:
    """
    Group rows by a combination of `SK_ID_CURR` and `MONTHS_BALANCE` based
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
    include_uniques : bool, optional
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
    >>>     },
    >>>     include_uniques=True
    >>> )
    """
    """ TODO virer après test de non régression
    # Load data
    data = get_table(table_name).copy()

    # Insert the aggregation counter
    data.insert(0, "n_PREV", 1)

    # Calculate or adjust MONTHS_BALANCE
    if months_balance_builder is None:
        data.MONTHS_BALANCE = -data.MONTHS_BALANCE
    else:
        data.insert(0, "MONTHS_BALANCE", months_balance_builder(data))

    if include_uniques:  # No distinction
        data_multis = data
    else:  # Protect singles
        # Split into sub-tables : single-PREV vs multi-PREV
        unique_prev_idx, _, _, _ = get_unique_and_multi_index(
            table_name, "SK_ID_PREV", "SK_ID_CURR"
        )
        is_single_mask = data.SK_ID_CURR.isin(unique_prev_idx.SK_ID_CURR)
        data_multis = data[~is_single_mask]
        data_uniques = data[is_single_mask]

    # Group multi-PREV by CURR-MONTH
    grouped = (
        data_multis.drop(columns="SK_ID_PREV")
        .groupby(by=["SK_ID_CURR", "MONTHS_BALANCE"])
    )

    # Aggregate data
    if agg_dict is None:
        grouped_multis = grouped.sum()
    if agg_dict is not None:
        _agg_dict = {"n_PREV": "sum"} | agg_dict
        grouped_multis = grouped.agg(_agg_dict)

    # Post-process `NAME_CONTRACT_STATUS` (if exists)
    if "NAME_CONTRACT_STATUS" in grouped_multis.columns:
        grouped_multis.NAME_CONTRACT_STATUS = (
            grouped_multis.NAME_CONTRACT_STATUS
            .apply(eval_contracts_status)
        )

    grouped_multis.reset_index(level=1, inplace=True)
    if include_uniques:
        return grouped_multis
    return _extracted_from_groupby_curr(
        data_uniques, grouped_multis, "MONTHS_BALANCE"
    )
    """
    return _groupby_curr_pivot(
        table_name, "MONTHS_BALANCE",
        months_balance_builder, agg_dict,
        include_uniques
    )


def groupby_curr_num(
    table_name: str,
    months_balance_builder: callable = None,
    agg_dict: dict = None,
    include_uniques: bool = False
) -> pd.DataFrame:
    """
    Group rows by a combination of `SK_ID_CURR` and `NUM_INSTALMENT_NUMBER` based
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
    include_uniques : bool, optional
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
    >>> grouped_data = groupby_curr_num(
    >>>     table_name="installments_payments",
    >>>     months_balance_builder=ip_months_balance_builder,
    >>>     agg_dict={
    >>>         "NUM_INSTALMENT_VERSION": "max",
    >>>         "DAYS_INSTALMENT": "median",
    >>>         "DAYS_ENTRY_PAYMENT": "median",
    >>>         "AMT_INSTALMENT": "sum",
    >>>         "AMT_PAYMENT": "sum"
    >>>     },
    >>>     include_uniques=True
    >>> )
    """
    """ TODO virer après test de non régression
    # Load data
    data = get_table(table_name).copy()

    # Insert the aggregation counter
    data.insert(0, "n_PREV", 1)

    # Calculate or adjust MONTHS_BALANCE
    if months_balance_builder is None:
        data.MONTHS_BALANCE = -data.MONTHS_BALANCE
    else:
        data.insert(0, "MONTHS_BALANCE", months_balance_builder(data))

    if include_uniques:  # No distinction
        data_multis = data
    else:  # Protect uniques
        # Split into sub-tables : single-PREV vs multi-PREV
        unique_prev_idx, _, _, _ = get_unique_and_multi_index(
            table_name, "SK_ID_PREV", "SK_ID_CURR"
        )
        is_single_mask = data.SK_ID_CURR.isin(unique_prev_idx.SK_ID_CURR)
        data_multis = data[~is_single_mask]
        data_uniques = data[is_single_mask]

    # Group multi-PREV by NUM_INSTALMENT_NUMBER
    grouped = (
        data_multis.drop(columns="SK_ID_PREV")
        .groupby(by=["SK_ID_CURR", "NUM_INSTALMENT_NUMBER"])
    )

    # Aggregate data
    if agg_dict is None:
        grouped_multis = grouped.sum()
    if agg_dict is not None:
        _agg_dict = {"n_PREV": "sum"} | agg_dict
        grouped_multis = grouped.agg(_agg_dict)

    # Post-process `NAME_CONTRACT_STATUS` (if exists)
    if "NAME_CONTRACT_STATUS" in grouped_multis.columns:
        grouped_multis.NAME_CONTRACT_STATUS = (
            grouped_multis.NAME_CONTRACT_STATUS
            .apply(eval_contracts_status)
        )

    grouped_multis.reset_index(level=1, inplace=True)
    if include_uniques:
        return grouped_multis
    return _extracted_from_groupby_curr(
        data_uniques, grouped_multis, "NUM_INSTALMENT_NUMBER"
    )
    """
    return _groupby_curr_pivot(
        table_name, "NUM_INSTALMENT_NUMBER",
        months_balance_builder, agg_dict,
        include_uniques
    )


