from typing import Callable, Optional, Union, List

from home_credit.load import get_table
from home_credit.merge import currentize
from pepper.feat_eng import jumps_rle, series_rle_reduction

import pandas as pd
import numpy as np


""" Bureau Balance v1
"""

"""
def get_currentized_bureau_balance() -> pd.DataFrame:
    ""
    DEPRECATED Use BureauBalance.clean() instead
    
    Currentize the 'bureau_balance' table by adding 'SK_ID_CURR' column
    and adjusting 'MONTHS_BALANCE'.

    Returns
    -------
    pd.DataFrame
        Currentized DataFrame with columns ['SK_ID_BUREAU', 'SK_ID_CURR', ...].
    ""
    # Load the 'bureau_balance' table
    data = get_table("bureau_balance").copy()

    # Add 'SK_ID_CURR' column by mapping 'SK_ID_BUREAU' to 'SK_ID_CURR'
    currentize(data)

    # Negate the 'MONTHS_BALANCE' column to ensure consistency
    data.MONTHS_BALANCE = -data.MONTHS_BALANCE

    # Remove records without a valid 'SK_ID_CURR' (not related to a client)
    return data.dropna()
"""

"""
def get_bureau_balance_loan_counts() -> pd.DataFrame:
    ""
    DEPRECATED Use get_bureau_balance_loan_counts_v2 instead
    
    Generate a table indicating the number of active loans for each client
    for each month.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ['SK_ID_CURR', 'MONTHS_BALANCE', 'ACTIVE_LOANS'].
    ""
    data = get_currentized_bureau_balance()

    # Filter out records with STATUS 'C' or 'X'
    scored_data = data[~data.STATUS.isin(["C", "X"])]

    # Group by 'SK_ID_CURR' and 'MONTHS_BALANCE' and count the number of records
    loan_counts = (
        scored_data.drop(columns="SK_ID_BUREAU")
        .groupby(by=["SK_ID_CURR", "MONTHS_BALANCE"])
        .agg("count")
        .rename(columns={"STATUS": "ACTIVE_LOANS"})
    )

    return loan_counts.reset_index()
"""

"""
def get_bureau_balance_loan_counts_v2() -> pd.DataFrame:
    ""
    DEPRECATED
    
    Generate a table indicating the number of active loans for each client
    for each month.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ['SK_ID_CURR', 'MONTHS_BALANCE', 'ACTIVE_LOANS'].
    ""
    data = get_currentized_bureau_balance()

    # Replace 'X' with NaN in the 'STATUS' column
    data.STATUS.replace("X", pd.NA, inplace=True)

    # Sort the data by 'SK_ID_BUREAU' and 'MONTHS_BALANCE'
    data.sort_values(by=["SK_ID_BUREAU", "MONTHS_BALANCE"], inplace=True)

    # Forward fill the missing values in 'ACTIVE_LOANS'
    data.STATUS.fillna(method="ffill", inplace=True)

    # Create a new column with 1 for active loans (or remaining NaNs) and 0 for 'C'
    data.STATUS = (data.STATUS != "C")

    # Group by 'SK_ID_CURR' and 'MONTHS_BALANCE' and sum the ACTIVE_LOANS values
    loan_counts = (
        data.drop(columns=["SK_ID_BUREAU"])
        .groupby(by=["SK_ID_CURR", "MONTHS_BALANCE"])
        .agg("sum")
        .rename(columns={"STATUS": "ACTIVE_LOANS"})
    )

    return loan_counts.reset_index()
"""

"""
def get_bureau_balance_composite_status() -> pd.DataFrame:
    ""
    DEPRECATED
    
    Generate a table with a composite STATUS for active loans.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ['SK_ID_CURR', 'MONTHS_BALANCE', 'COMPOSITE_STATUS'].
    ""
    data = get_currentized_bureau_balance()

    # Filter out records with STATUS 'C' or 'X'
    scored_data = data[~data.STATUS.isin(["C", "X"])].copy()

    # Create a new column with STATUS values exponentiated
    scored_data.STATUS = np.exp(scored_data.STATUS.astype(int))

    # Group by 'SK_ID_CURR' and 'MONTHS_BALANCE' and sum the modified STATUS values
    agg_status = (
        scored_data.drop(columns="SK_ID_BUREAU")
        .groupby(by=["SK_ID_CURR", "MONTHS_BALANCE"])
        .agg("sum")
    )

    # Apply the final log transformation to the composite STATUS
    agg_status["COMPOSITE_STATUS"] = np.log(agg_status.STATUS)

    # Keep only the relevant columns and reset the index
    return agg_status[["COMPOSITE_STATUS"]].reset_index()
"""

"""
def get_bureau_balance_composite_status_v2() -> pd.DataFrame:
    ""
    DEPERECATED Use aggregate_loan_status_by_loan instead
    
    Generate a table with a composite STATUS for active loans.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ['SK_ID_CURR', 'MONTHS_BALANCE', 'COMPOSITE_STATUS'].
    ""
    data = get_currentized_bureau_balance()
    
    # Replace 'C' with 0 in the 'STATUS' column
    data.STATUS.replace("C", 0, inplace=True)
    
    # Replace 'X' with NaN in the 'STATUS' column
    data.STATUS.replace("X", pd.NA, inplace=True)

    # Sort the data by 'SK_ID_BUREAU' and 'MONTHS_BALANCE'
    data.sort_values(by=["SK_ID_BUREAU", "MONTHS_BALANCE"], inplace=True)

    # Forward fill the missing values in 'STATUS'
    data.STATUS.fillna(method='ffill', inplace=True)

    # Create a new column with STATUS values exponentiated
    data.STATUS = np.exp(data.STATUS.astype(int))

    # Group by 'SK_ID_CURR' and 'MONTHS_BALANCE' and sum the modified STATUS values
    agg_status = (
        data.drop(columns="SK_ID_BUREAU")
        .groupby(by=["SK_ID_CURR", "MONTHS_BALANCE"])
        .agg("sum")
    )

    # Apply the final log transformation to the composite STATUS
    agg_status["COMPOSITE_STATUS"] = np.round(np.log(agg_status.STATUS), 1)

    # Keep only the relevant columns and reset the index
    return agg_status[["COMPOSITE_STATUS"]].reset_index()
"""


"""
def process_rle_variations(data, month_col, variation_col):
    ""DEPRECATED""
    grouped = data.groupby(by=["SK_ID_CURR"])
    variations = grouped.agg({
        month_col: [min, max, len, jumps_rle],
        variation_col: series_rle_reduction,
    })
    variations.columns = [
        "months_min", "months_max", "months_len", "months_jumps",
        f"{variation_col}_variation"
    ]
    return variations.reset_index()
"""

"""
def get_loan_counts_variation():
    ""DEPRECATED""
    loan_counts = get_bureau_balance_loan_counts_v2()
    return process_rle_variations(loan_counts, "MONTHS_BALANCE", "ACTIVE_LOANS")
"""

"""
def get_composite_status_variation():
    ""DEPRECATED""
    composite_status = get_bureau_balance_composite_status_v2()
    return process_rle_variations(composite_status, "MONTHS_BALANCE", "COMPOSITE_STATUS")
"""

"""
def join_variations_tables():
    ""DEPRECATED""
    loan_counts_variation = get_loan_counts_variation()
    composite_status_variation = get_composite_status_variation()

    return loan_counts_variation.merge(
        composite_status_variation.drop(
            columns=[
                col
                for col in composite_status_variation.columns
                if col.startswith("months_")
            ]
        ),
        on="SK_ID_CURR",
        how="inner",  # You can adjust the merge type as needed
    )
"""


""" Bureau Balance v2
"""

def aggregate_loan_status(
    data: pd.DataFrame,
    pivot_columns: Union[str, List[str]],
    data_preprocessor: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    dropped_columns: Optional[Union[str, List[str]]] = None,
    alpha: Optional[float] = 1,
    decimals: Optional[int] = 2,
    result_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregate loan status data by given pivot columns.

    This function aggregates loan status data by grouping it
    using one or more pivot columns.
    
    It provides options for customizing data preprocessing,
    column exclusion, and the application of amortization
    to the combined 'STATUS' risk factor.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing loan status data.
    pivot_columns : Union[str, List[str]]
        The column(s) to use as pivot for grouping.
    data_preprocessor : Callable, optional
        A function to preprocess the data before aggregation, by default None.
    dropped_columns : Union[str, List[str]], optional
        Columns to be dropped from the input data, by default None.
    alpha : float, optional
        A parameter for amortization, by default 1.
    decimals : int, optional
        The number of decimal places for rounding, by default 2.

    Returns
    -------
    pd.DataFrame
        A DataFrame with aggregated loan status information,
        including pivot columns and 'STATUS'.
    """
    aggregated = data_preprocessor(data) if data_preprocessor else data
    
    aggregated = aggregated.reset_index()

    # Drop the irrelevant columns
    if dropped_columns:
        aggregated = aggregated.drop(columns=dropped_columns)
    
    # Encode STATUS
    aggregated.STATUS = aggregated.STATUS.replace("C", 0)
    aggregated.STATUS = aggregated.STATUS.astype(np.uint8)

    # Exponentiate STATUS
    status = aggregated.STATUS
    aggregated.STATUS = np.exp(status)
    
    # Amortize STATUS if required (alpha != 0)
    if alpha:
        status = aggregated.STATUS
        months = aggregated.MONTHS_BALANCE
        aggregated.STATUS = status / (1 + alpha * months)

    # Group by the pivot columns
    aggregated = aggregated.groupby(by=pivot_columns)
    
    # Sum and count the modified STATUS values
    aggregated = aggregated.agg({"STATUS": ["sum", "count"]})

    # Apply the final log transformation to the composite STATUS
    exp_status_sum = aggregated[("STATUS", "sum")]
    exp_status_count = aggregated[("STATUS", "count")]
    exp_status = exp_status_sum / exp_status_count
    aggregated["agg_S"] = np.round(np.log(exp_status), decimals)

    # Keep the only relevant columns
    aggregated = aggregated[["agg_S"]]

    # Reset column names to the first level
    aggregated.columns = aggregated.columns.get_level_values(0)

    # Rename the 'agg_S' column to 'STATUS'
    aggregated.rename(columns={"agg_S": "STATUS"}, inplace=True)

    aggregated.STATUS = aggregated.STATUS.astype(np.float32)
    
    if result_name:
        aggregated.columns.name = result_name

    return aggregated


def get_bureau_loan_status_by_month(
    data: pd.DataFrame,
    decimals: Optional[int] = 2
) -> pd.DataFrame:
    """
    Aggregate loan status data by 'SK_ID_BUREAU' and 'MONTHS_BALANCE'.

    This function aggregates loan status data by grouping it
    using 'SK_ID_BUREAU' and 'MONTHS_BALANCE'.
    
    It computes the composite status as the sum of exp(Status)
    divided by the count of records for each group,
    applies a logarithmic transformation,
    and returns the result in a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing loan status data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with aggregated loan status information,
        including 'SK_ID_BUREAU', 'MONTHS_BALANCE', and 'COMPOSITE_STATUS'.

    Notes
    -----
    Here, do not perform an aggregation that returns a single element per group.
    It is essentially an expensive identity operation that would be impractical
    to encapsulate in a library. DO NOT THIS :
    >>> return aggregate_loan_status(
    >>>     data=data,
    >>>     pivot_columns=["SK_ID_BUREAU", "MONTHS_BALANCE"],
    >>>     data_preprocessor=None,
    >>>     dropped_columns=["TARGET", "SK_ID_CURR"],
    >>>     alpha=0,
    >>>     decimals=decimals,
    >>>     result_name="BUREAU_LOAN_STATUS_BY_MONTH"
    >>> )
    """
    status = data[["STATUS"]].copy()
    status.STATUS = status.STATUS.replace("C", 0)
    status.STATUS = status.STATUS.astype(np.uint8)
    status.columns.name = "BUREAU_LOAN_STATUS_BY_MONTH"
    return status


def get_bureau_loan_status_by_client_and_month(
    data: pd.DataFrame,
    decimals: Optional[int] = 2
) -> pd.DataFrame:
    """
    Aggregate loan status data by 'SK_ID_CURR' and 'MONTHS_BALANCE'.

    This function aggregates loan status data by grouping it
    using 'SK_ID_CURR' and 'MONTHS_BALANCE'.
    
    It computes the composite status as the sum of exp(Status)
    divided by the count of records for each group,
    applies a logarithmic transformation,
    and returns the result in a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing loan status data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with aggregated loan status information,
        including 'SK_ID_CURR', 'MONTHS_BALANCE', and 'STATUS'.
    """
    return aggregate_loan_status(
        data=data,
        pivot_columns=["SK_ID_CURR", "MONTHS_BALANCE"],
        data_preprocessor=None,
        dropped_columns=["TARGET", "SK_ID_BUREAU"],
        alpha=0,
        decimals=decimals,
        result_name="BUREAU_LOAN_STATUS_BY_CLIENT_AND_MONTH"
    )


def get_bureau_loan_status(
    data: pd.DataFrame,
    alpha: Optional[float] = 1,
    decimals: Optional[int] = 2
) -> pd.DataFrame:
    return aggregate_loan_status(
        data=data,
        pivot_columns=["SK_ID_BUREAU"],
        data_preprocessor=None,
        dropped_columns=["TARGET", "SK_ID_CURR"],
        alpha=alpha,
        decimals=decimals,
        result_name="BUREAU_LOAN_STATUS"
    )


def get_bureau_loan_status_by_client(
    data: pd.DataFrame,
    alpha: Optional[float] = 1,
    decimals: Optional[int] = 2
) -> pd.DataFrame:
    return aggregate_loan_status(
        data=data,
        pivot_columns=["SK_ID_CURR"],
        data_preprocessor=get_bureau_loan_status_by_client_and_month,
        dropped_columns=None,
        alpha=alpha,
        decimals=decimals,
        result_name="BUREAU_LOAN_STATUS_BY_CLIENT"
    )


def get_bureau_loan_activity_by_month(data: pd.DataFrame, keep_curr=False):
    dropped_cols = ["TARGET"] if keep_curr else ["TARGET", "SK_ID_CURR"]
    activity = data.drop(columns=dropped_cols).copy()
    activity.STATUS = (activity.STATUS != "C").astype(np.uint8)
    activity.rename(columns={"STATUS": "ACTIVE"}, inplace=True)
    activity.columns.name = "BUREAU_LOAN_ACTIVITY_BY_MONTH"
    return activity


def get_bureau_loan_activity_by_client_and_month(data):
    activity = get_bureau_loan_activity_by_month(data, keep_curr=True)
    activity = activity.reset_index()
    activity = activity.drop(columns="SK_ID_BUREAU")
    activity = activity.groupby(by=["SK_ID_CURR", "MONTHS_BALANCE"])
    activity = activity.agg("sum")
    activity.ACTIVE = activity.ACTIVE.astype(np.uint8)
    activity.columns.name = "BUREAU_LOAN_ACTIVITY_BY_CLIENT_AND_MONTH"
    return activity


def get_bureau_mean_loan_activity(data, alpha=1):
    activity = get_bureau_loan_activity_by_month(data)
    activity = activity.reset_index()
    # Amortize STATUS if required (alpha != 0)
    if alpha:
        active = activity.ACTIVE
        months = activity.MONTHS_BALANCE
        activity.ACTIVE = active / (1 + alpha * months)
    activity = activity.drop(columns="MONTHS_BALANCE")
    activity = activity.groupby(by="SK_ID_BUREAU")
    activity = activity.agg("mean")
    activity.ACTIVE = activity.ACTIVE.astype(np.float32)
    activity.columns.name = "BUREAU_MEAN_LOAN_ACTIVITY"
    return activity


def get_bureau_mean_loan_activity_by_client(data, alpha=1):
    activity = get_bureau_loan_activity_by_client_and_month(data)
    activity = activity.reset_index()
    # Amortize STATUS if required (alpha != 0)
    if alpha:
        active = activity.ACTIVE
        months = activity.MONTHS_BALANCE
        activity.ACTIVE = active / (1 + alpha * months)
    activity = activity.drop(columns="MONTHS_BALANCE")
    activity = activity.groupby(by="SK_ID_CURR")
    activity = activity.agg("mean")
    activity.ACTIVE = activity.ACTIVE.astype(np.float32)
    activity.columns.name = "BUREAU_MEAN_LOAN_ACTIVITY_BY_CLIENT"
    return activity


def get_rle_bureau_loan_tracking_period(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate tracking periods for loans using run-length encoding.

    This function calculates the tracking periods for loans by grouping
    the data by 'SK_ID_BUREAU' and applying run-length encoding (RLE)
    to the 'MONTHS_BALANCE' column. The RLE algorithm identifies runs
    of consecutive months and returns the start and end months for each run.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing loan tracking data with columns
        'SK_ID_BUREAU' and 'MONTHS_BALANCE'.

    Returns
    -------
    pd.DataFrame
        A DataFrame with tracking periods for each loan,
        including 'SK_ID_BUREAU', 'min', 'max', 'count', and 'jumps_rle'.
    """
    tracking = data.reset_index()
    tracking = tracking[["SK_ID_BUREAU", "MONTHS_BALANCE"]]
    # tracking.SK_ID_BUREAU = tracking.SK_ID_BUREAU.astype(np.uint32)
    tracking.MONTHS_BALANCE = tracking.MONTHS_BALANCE.astype(np.uint8)
    tracking = tracking.groupby(by="SK_ID_BUREAU")
    tracking = tracking.agg({"MONTHS_BALANCE": ["min", "max", "count", jumps_rle]})
    months_agg_cols = tracking.columns[:3]
    tracking[months_agg_cols] = tracking[months_agg_cols].astype(np.uint8)
    tracking.columns.name = "BUREAU_LOAN_TRACKING_PERIOD"
    return tracking


def get_rle_bureau_loan_tracking_period_by_client(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate tracking periods for applicants using run-length encoding.

    This function calculates the tracking periods for applicants by sorting
    the data by 'SK_ID_CURR' and 'MONTHS_BALANCE', removing duplicates,
    and applying run-length encoding (RLE) to the 'MONTHS_BALANCE' column.
    The RLE algorithm identifies runs of consecutive months and returns
    the start and end months for each run.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing applicant tracking data with columns
        'SK_ID_CURR' and 'MONTHS_BALANCE'.

    Returns
    -------
    pd.DataFrame
        A DataFrame with tracking periods for each applicant,
        including 'SK_ID_CURR', 'min', 'max', 'count', and 'jumps_rle'.
    """
    tracking = data.reset_index()
    tracking = tracking[["SK_ID_CURR", "MONTHS_BALANCE"]]
    # tracking.SK_ID_CURR = tracking.SK_ID_CURR.astype(np.uint32)
    tracking.MONTHS_BALANCE = tracking.MONTHS_BALANCE.astype(np.uint8)
    tracking = tracking.sort_values(by=["SK_ID_CURR", "MONTHS_BALANCE"])
    tracking = tracking.drop_duplicates()
    tracking = tracking.groupby(by="SK_ID_CURR")
    tracking = tracking.agg({"MONTHS_BALANCE": ["min", "max", "count", jumps_rle]})
    months_agg_cols = tracking.columns[:3]
    tracking[months_agg_cols] = tracking[months_agg_cols].astype(np.uint8)
    tracking.columns.name = "BUREAU_LOAN_TRACKING_PERIOD_BY_CLIENT"
    return tracking


def get_rle_bureau_loan_feature_variations(
    data: pd.DataFrame,
    features: Union[str, List[str]]
) -> pd.DataFrame:
    """
    Calculate run-length encoded (RLE) variations for specified features per loan.

    This function computes RLE variations for specified features of loans based on
    the 'SK_ID_BUREAU' identifier. It provides flexibility to calculate RLE variations
    for one or more features.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing loan data.
    features : Union[str, List[str]]
        A single feature or a list of features for which RLE variations will be calculated.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing RLE variations for each specified feature per loan,
        with columns for 'SK_ID_BUREAU', 'M_1ST', 'M_LST', 'M_CNT', 'M_RLE', and the specified features.

    Example
    -------
    >>> data = pd.DataFrame({'SK_ID_BUREAU': [1, 1, 1, 2, 2, 2],
    ...                      'MONTHS_BALANCE': [0, 1, 2, 0, 1, 2],
    ...                      'FEATURE_A': [10, 10, 15, 5, 5, 5],
    ...                      'FEATURE_B': [0, 1, 0, 1, 1, 0]})
    >>> get_rle_loan_feature_variations(data, ['FEATURE_A', 'FEATURE_B'])
    
       SK_ID_BUREAU  M_1ST  M_LST  M_CNT       M_RLE  FEATURE_A  FEATURE_B
    0             1      0      2      3  ((10, 2), (15, 1))         10          0
    1             2      0      2      3    ((5, 3), (1, 2))          5          1
    """
    
    if isinstance(features, str):
        features = [features]
    
    tracking = data.reset_index()
    tracking = tracking[["SK_ID_BUREAU", "MONTHS_BALANCE"] + features]
    # tracking.SK_ID_BUREAU = tracking.SK_ID_BUREAU.astype(np.uint32)
    tracking.MONTHS_BALANCE = tracking.MONTHS_BALANCE.astype(np.uint8)
    tracking = tracking.groupby(by="SK_ID_BUREAU")
    
    agg_rules = (
        {"MONTHS_BALANCE": ["min", "max", "count", jumps_rle]} |
        {feature: series_rle_reduction for feature in features}
    )
    
    tracking = tracking.agg(agg_rules)

    months_agg_cols = tracking.columns[:3]
    tracking[months_agg_cols] = tracking[months_agg_cols].astype(np.uint8)
    tracking.columns.name = f"{data.columns.name}_VARIATIONS"
    
    return tracking


def get_rle_bureau_loan_feature_by_client_variations(
    data: pd.DataFrame,
    features: Union[str, List[str]]
) -> pd.DataFrame:
    """
    Calculate run-length encoded (RLE) variations for specified features per applicant.

    This function computes RLE variations for specified features of applicants based on
    the 'SK_ID_CURR' identifier. It allows for flexibility in calculating RLE variations
    for one or more features.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing applicant data.
    features : Union[str, List[str]]
        A single feature or a list of features for which RLE variations will be calculated.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing RLE variations for each specified feature per applicant,
        with columns for 'SK_ID_CURR', 'M_1ST', 'M_LST', 'M_CNT', 'M_RLE', and the specified features.

    Example
    -------
    >>> data = pd.DataFrame({'SK_ID_CURR': [1, 1, 1, 2, 2, 2],
    ...                      'MONTHS_BALANCE': [0, 1, 2, 0, 1, 2],
    ...                      'FEATURE_A': [10, 10, 15, 5, 5, 5],
    ...                      'FEATURE_B': [0, 1, 0, 1, 1, 0]})
    >>> get_rle_applicant_feature_variations(data, ['FEATURE_A', 'FEATURE_B'])
    
       SK_ID_CURR  M_1ST  M_LST  M_CNT       M_RLE  FEATURE_A  FEATURE_B
    0           1      0      2      3  ((10, 2), (15, 1))         10          0
    1           2      0      2      3    ((5, 3), (1, 2))          5          1
    """
    
    if isinstance(features, str):
        features = [features]
    
    tracking = data.reset_index()
    tracking = tracking[["SK_ID_CURR", "MONTHS_BALANCE"] + features]
    # tracking.SK_ID_CURR = tracking.SK_ID_CURR.astype(np.uint32)
    tracking.MONTHS_BALANCE = tracking.MONTHS_BALANCE.astype(np.uint8)
    tracking = tracking.sort_values(by=["SK_ID_CURR", "MONTHS_BALANCE"])    
    tracking = tracking.groupby(by="SK_ID_CURR")
    
    agg_rules = (
        {"MONTHS_BALANCE": ["min", "max", "count", jumps_rle]} |
        {feature: series_rle_reduction for feature in features}
    )
    
    tracking = tracking.agg(agg_rules)

    months_agg_cols = tracking.columns[:3]
    tracking[months_agg_cols] = tracking[months_agg_cols].astype(np.uint8)
    tracking.columns.name = f"{data.columns.name}_BY_CLIENT_VARIATIONS"
    
    return tracking


""" Bureau
"""


def get_extended_clean_bureau(
    include_status: bool = True,
    include_activity: bool = True,
    include_status_variation: bool = False,
    include_activity_variation: bool = False
) -> pd.DataFrame:
    pass


def get_extended_clean_bureau_by_client():
    pass


""" Previous application
"""


def get_extended_clean_previous_application(
    include_status: bool = True,
    include_activity: bool = True,
    include_status_variation: bool = False,
    include_activity_variation: bool = False
) -> pd.DataFrame:
    pass


def get_extended_clean_previous_application_by_client():
    pass
