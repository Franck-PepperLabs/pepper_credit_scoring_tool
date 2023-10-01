from typing import Callable, Optional, Union, List

from home_credit.load import get_table
from home_credit.merge import currentize
from pepper.np_utils import subindex_nd
from pepper.rle import jumps_rle, series_rle_reduction

import pandas as pd
import numpy as np



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
    aggregated.STATUS = aggregated.STATUS.replace("C", 0).astype(np.uint8)
    # aggregated.STATUS = aggregated.STATUS.astype(np.uint8)

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

    aggregated.STATUS = aggregated.STATUS.astype(np.float64)
    
    if result_name:
        aggregated.columns.name = result_name

    return aggregated


def aggregate_loan_activity(
    data: pd.DataFrame,
    pivot_columns: Union[str, List[str]],
    data_preprocessor: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    dropped_columns: Optional[Union[str, List[str]]] = None,
    alpha: Optional[float] = 1,
    decimals: Optional[int] = 2,
    result_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregate loan activity data by given pivot columns.

    This function aggregates loan activity data by grouping it
    using one or more pivot columns.
    
    It provides options for customizing data preprocessing,
    column exclusion, and the application of amortization
    to the combined 'ACTIVITY' indicator.

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
        A DataFrame with aggregated loan activity information,
        including pivot columns and 'ACTIVITY'.
    """
    aggregated = data_preprocessor(data) if data_preprocessor else data
    
    aggregated = aggregated.reset_index()

    # Drop the irrelevant columns
    if dropped_columns:
        aggregated = aggregated.drop(columns=dropped_columns)
    
    # Encode STATUS
    # aggregated.STATUS = aggregated.STATUS.replace("C", 0)
    # aggregated.STATUS = aggregated.STATUS.astype(np.uint8)
    aggregated.STATUS = (aggregated.STATUS != "C").astype(np.uint8)
    aggregated.rename(columns={"STATUS": "ACTIVITY"}, inplace=True)
    
    # Exponentiate STATUS
    # status = aggregated.STATUS
    # aggregated.STATUS = np.exp(status)
    
    # Amortize ACTIVITY if required (alpha != 0)
    if alpha:
        activity = aggregated.ACTIVITY
        months = aggregated.MONTHS_BALANCE
        aggregated.ACTIVITY = activity / (1 + alpha * months)

    # Group by the pivot columns
    aggregated = aggregated.groupby(by=pivot_columns)
    
    # Sum and count the modified ACTIVITY values
    aggregated = aggregated.agg({"ACTIVITY": ["sum", "count"]})

    # Apply the final log transformation to the composite ACTIVITY
    activity_sum = aggregated[("ACTIVITY", "sum")]
    activity_count = aggregated[("ACTIVITY", "count")]
    activity = activity_sum / activity_count
    aggregated["agg_S"] = np.round(activity, decimals)

    # Keep the only relevant columns
    aggregated = aggregated[["agg_S"]]

    # Reset column names to the first level
    aggregated.columns = aggregated.columns.get_level_values(0)

    # Rename the 'agg_S' column to 'ACTIVITY'
    aggregated.rename(columns={"agg_S": "ACTIVITY"}, inplace=True)

    aggregated.ACTIVITY = aggregated.ACTIVITY.astype(np.float64)
    
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


def get_bureau_loan_activity(
    data: pd.DataFrame,
    alpha: Optional[float] = 1,
    decimals: Optional[int] = 2
) -> pd.DataFrame:
    return aggregate_loan_activity(
        data=data,
        pivot_columns=["SK_ID_BUREAU"],
        data_preprocessor=None,
        dropped_columns=["TARGET", "SK_ID_CURR"],
        alpha=alpha,
        decimals=decimals,
        result_name="BUREAU_LOAN_ACTIVITY"
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


def get_bureau_loan_activity_by_month(data: pd.DataFrame, keep_curr=True):
    dropped_cols = ["TARGET"] if keep_curr else ["TARGET", "SK_ID_CURR"]
    activity = data.drop(columns=dropped_cols).copy()
    activity.STATUS = (activity.STATUS != "C").astype(np.uint8)
    activity.rename(columns={"STATUS": "ACTIVITY"}, inplace=True)
    activity.columns.name = "BUREAU_LOAN_ACTIVITY_BY_MONTH"
    return activity


def get_bureau_loan_activity_by_client_and_month(data):
    activity = get_bureau_loan_activity_by_month(data)
    activity = activity.reset_index()
    activity = activity.drop(columns="SK_ID_BUREAU")
    activity = activity.groupby(by=["SK_ID_CURR", "MONTHS_BALANCE"])
    activity = activity.agg("sum")
    activity.ACTIVITY = activity.ACTIVITY.astype(np.uint8)
    activity.columns.name = "BUREAU_LOAN_ACTIVITY_BY_CLIENT_AND_MONTH"
    return activity


def get_bureau_mean_loan_activity(data, alpha=1):
    activity = get_bureau_loan_activity_by_month(data)
    activity = activity.reset_index()
    # Amortize STATUS if required (alpha != 0)
    if alpha:
        active = activity.ACTIVITY
        months = activity.MONTHS_BALANCE
        activity.ACTIVITY = active / (1 + alpha * months)
    activity = activity.drop(columns=["SK_ID_CURR", "MONTHS_BALANCE"])
    activity = activity.groupby(by="SK_ID_BUREAU")
    activity = activity.agg("mean")
    activity.ACTIVITY = activity.ACTIVITY.astype(np.float32)
    activity.columns.name = "BUREAU_MEAN_LOAN_ACTIVITY"
    return activity


def get_bureau_mean_loan_activity_by_client(data, alpha=1):
    activity = get_bureau_loan_activity_by_client_and_month(data)
    activity = activity.reset_index()
    # Amortize STATUS if required (alpha != 0)
    if alpha:
        active = activity.ACTIVITY
        months = activity.MONTHS_BALANCE
        activity.ACTIVITY = active / (1 + alpha * months)
    activity = activity.drop(columns="MONTHS_BALANCE")
    activity = activity.groupby(by="SK_ID_CURR")
    activity = activity.agg("mean")
    activity.ACTIVITY = activity.ACTIVITY.astype(np.float32)
    activity.columns.name = "BUREAU_MEAN_LOAN_ACTIVITY_BY_CLIENT"
    return activity


def get_rle_tracking_period(
    data: pd.DataFrame,
    pivot: str,
    base_name: str
) -> pd.DataFrame:
    """
    Get Run-Length Encoding (RLE) tracking periods for a given DataFrame.

    This function extracts tracking periods from the input DataFrame
    based on the specified pivot column.
    
    The tracking periods are calculated using the 'jumps_rle' function
    and returned in a new DataFrame.

    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame containing data to analyze.

    pivot : str
        The name of the pivot column used for grouping and tracking.

    base_name : str
        A base name used to create the column name for the result DataFrame.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing RLE tracking periods for each group defined
        by the pivot column.

    Notes:
    ------
    - The 'date_col' column is converted to arrays as it contains tuples.
    - Post-conversion to ndarray is necessary due to the tuple returned
        by the 'jumps_rle' function. 'jumps_rle' cannot directly return
        an ndarray; otherwise, aggregation would fail
        with a ValueError: Must produce aggregated value.
    """
    # Define the date column
    date_col = "MONTHS_BALANCE"
    
    # Reset the index of the input data for further processing
    tracking = data.reset_index()
    
    # Select relevant columns for tracking and sort them
    tracking = tracking[[pivot, date_col]]
    tracking = tracking.sort_values(by=[pivot, date_col])
    
    # Remove duplicate rows based on the selected columns
    tracking = tracking.drop_duplicates()
    
    # Convert the 'date_col' column to an unsigned 8-bit integer
    tracking[date_col] = tracking[date_col].astype(np.uint8)
    
    # Group the data by the specified 'pivot' column
    tracking = tracking.groupby(by=pivot)

    # Note: Post-conversion to ndarray is necessary due to the tuple returned
    # by the 'jumps_rle' function. 'jumps_rle' cannot directly return an ndarray;
    # otherwise, aggregation would fail with a ValueError: Must produce aggregated value.
    
    # Apply the 'jumps_rle' function for aggregation, which returns a tuple
    tracking = tracking.agg(jumps_rle)

    # Convert the 'date_col' column to arrays, as it contains tuples
    tracking[date_col] = tracking[date_col].apply(np.array)
    
    # Set the column name for the result DataFrame
    tracking.columns.name = f"{base_name}_TRACKING_PERIOD"
    
    return tracking


def get_rle_bureau_loan_feature_by_client_variation(
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
    tracking.MONTHS_BALANCE = tracking.MONTHS_BALANCE.astype(np.uint8)
    tracking = tracking.sort_values(by=["SK_ID_CURR", "MONTHS_BALANCE"])    
    tracking = tracking.groupby(by="SK_ID_CURR")
    agg_rules = {feature: series_rle_reduction for feature in features}
    tracking = tracking.agg(agg_rules)
    tracking[features] = tracking[features].applymap(np.array)
    tracking.columns.name = f"{data.columns.name}_BY_CLIENT_VARIATIONS"
    return tracking

def get_rle_bureau_loan_feature_variation(
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
        with columns for 'SK_ID_BUREAU', 'M_1ST', 'M_LST', 'M_CNT', 'M_RLE',
        and the specified features.

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
    tracking.MONTHS_BALANCE = tracking.MONTHS_BALANCE.astype(np.uint8)
    tracking = tracking.groupby(by="SK_ID_BUREAU")
    agg_rules = {feature: series_rle_reduction for feature in features}
    tracking = tracking.agg(agg_rules)
    tracking[features] = tracking[features].applymap(np.array)
    tracking.columns.name = f"{data.columns.name}_VARIATIONS"
    return tracking


def get_rle_pos_cash_loan_tracking_period(data: pd.DataFrame) -> pd.DataFrame:
    return get_rle_tracking_period(data, "SK_ID_PREV", "POS_CASH_LOAN")


def get_rle_credit_card_loan_tracking_period(data: pd.DataFrame) -> pd.DataFrame:
    return get_rle_tracking_period(data, "SK_ID_PREV", "CREDIT_CARD_LOAN")


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
    return get_rle_tracking_period(data, "SK_ID_BUREAU", "BUREAU_LOAN")


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
    return get_rle_tracking_period(data, "SK_ID_CURR", "BUREAU_LOAN_BY_CLIENT")


""" Bureau
"""


def get_extended_clean_bureau(
    data: pd.DataFrame,
    status: pd.DataFrame | None = None,
    activity: pd.DataFrame | None = None,
    tracking_period: pd.DataFrame | None = None,
    status_variation: pd.DataFrame | None = None,
    activity_variation: pd.DataFrame | None = None,
    how: str = "left"
) -> pd.DataFrame:
    """
    Extend and clean the 'data' DataFrame by merging it with optional additional DataFrames.

    This function takes a base DataFrame 'data' and optionally additional DataFrames
    (e.g., 'status', 'activity', 'tracking_period', 'status_variation', 'activity_variation')
    and extends the 'data' DataFrame by merging them based on a common column ('SK_ID_BUREAU').

    Parameters
    ----------
    data : pd.DataFrame
        The base DataFrame to be extended and cleaned.
    status : pd.DataFrame or None, optional
        An additional DataFrame containing status information (default is None).
    activity : pd.DataFrame or None, optional
        An additional DataFrame containing activity information (default is None).
    tracking_period : pd.DataFrame or None, optional
        An additional DataFrame containing tracking period information (default is None).
    status_variation : pd.DataFrame or None, optional
        An additional DataFrame containing status variation information (default is None).
    activity_variation : pd.DataFrame or None, optional
        An additional DataFrame containing activity variation information (default is None).
    how : str, optional
        The type of merge to be performed (default is 'left').

    Returns
    -------
    pd.DataFrame
        The extended and cleaned DataFrame resulting from merging 'data' with the optional
        additional DataFrames.

    Notes
    -----
    This function allows you to extend the 'data' DataFrame by merging it with other DataFrames
    based on a common column ('SK_ID_BUREAU'). If no additional DataFrames are provided, the
    function returns the original 'data' DataFrame.

    """
    # Combine provided DataFrames based on 'SK_ID_BUREAU' and 'how' parameter
    adj_data_parts = [
        activity, status,
        tracking_period, activity_variation, status_variation
    ]
    adj_data_parts = list(filter(lambda x: x is not None, adj_data_parts))
    if not adj_data_parts:
        return data

    adj_data = pd.concat(adj_data_parts, axis=1)
    on_and_how = {"on": "SK_ID_BUREAU", "how": how}
    data = pd.merge(left=data.reset_index(), right=adj_data, **on_and_how)
    return data.set_index(["SK_ID_CURR", "SK_ID_BUREAU"])

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


""" Installments Payments
"""

def get_installments_payments_by_installment_and_version(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate installment payment data by installment and version.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing installment payment data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with aggregated installment payment information, including
        the days of installment, the installment amount, the count of payments,
        the start and end days of payment, and the total payment amount.

    Notes
    -----
    This function aggregates installment payment data based on the installment
    number and version, summarizing payment-related information.
    """
    # Select relevant columns for aggregation
    data = data[[
        "NUM_INSTALMENT_VERSION",
        "AMT_INSTALMENT", "AMT_PAYMENT",
        "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"
    ]].reset_index()

    # Sort the data for easier aggregation
    data = data.sort_values(by=[
        "SK_ID_CURR", "SK_ID_PREV",
        "NUM_INSTALMENT_NUMBER", "NUM_INSTALMENT_VERSION",
        "DAYS_ENTRY_PAYMENT"
    ])
    
    # Group the data for aggregation
    data = data.groupby(by=[
        "SK_ID_CURR", "SK_ID_PREV",
        "NUM_INSTALMENT_NUMBER", "NUM_INSTALMENT_VERSION"
    ])

    # Perform aggregation
    data = data.agg({
        "DAYS_INSTALMENT": "first",
        "AMT_INSTALMENT": "first",
        "DAYS_ENTRY_PAYMENT": ["count", "last", "first"],
        "AMT_PAYMENT": "sum"
    })

    # Rename the aggregated columns
    data.columns = [
        "DAYS_INSTALMENT", "AMT_INSTALMENT", "CNT_PAYMENT",
        "DAYS_ENTRY_PAYMENT_START", "DAYS_ENTRY_PAYMENT_END",
        "AMT_PAYMENT"
    ]
    
    return data


def get_clean_installments_payments_loan_amount(
    data: pd.DataFrame
) -> pd.DataFrame:
    grouped = data.groupby(by=["SK_ID_CURR", "SK_ID_PREV"])
    loan_amount = (
        grouped.agg({"AMT_INSTALMENT": "sum"})
        .rename(columns={"AMT_INSTALMENT": "AMT_LOAN"})
    )
    loan_amount.AMT_LOAN = loan_amount.AMT_LOAN.round(2)
    loan_amount.columns.name = "LOAN_AMOUNT"
    return loan_amount


def get_clean_installments_payments_loaned_amount(
    loan_amount: pd.DataFrame
) -> pd.DataFrame:
    data = loan_amount.groupby(by="SK_ID_CURR").agg("sum")
    data = data.rename(columns={"AMT_LOAN": "AMT_LOANED"})
    data.columns.name = "LOANED_AMOUNT"
    return data


def get_clean_installments_payments_last_version(
    data: pd.DataFrame
) -> pd.DataFrame:
    data = data.reset_index(level="NUM_INSTALMENT_NUMBER")
    data.columns = ["N", "M", "T", "V", "T0", "T1", "INST", "REPAID"]
    data = data[["M", "N", "V", "T0", "T1", "INST", "REPAID"]]
    data = data.reset_index()
    data = data.sort_values(by=["SK_ID_CURR", "SK_ID_PREV", "N", "V"])
    data_keys = data[["SK_ID_CURR", "SK_ID_PREV", "N", "V"]]
    data.insert(4, "n", subindex_nd(data_keys.to_numpy(), sorted=True))
    data = data.sort_values(by=["SK_ID_CURR", "SK_ID_PREV", "N", "n", "V"])
    keep_index = data[["SK_ID_CURR", "SK_ID_PREV", "N", "V"]]
    keep_index = keep_index.drop_duplicates(
        subset=["SK_ID_CURR", "SK_ID_PREV", "N"],
        keep="last"
    )
    data = data.set_index(["SK_ID_CURR", "SK_ID_PREV", "N", "V"])
    keep_index = pd.MultiIndex.from_frame(keep_index)
    return data[data.index.isin(keep_index)]


def get_clean_installments_payments_repaid_and_dpd(
    data: pd.DataFrame 
) -> pd.DataFrame:
    data = data.copy()
    data["DPD"] = data.T1 - data.T0
    data = data.groupby(by=["SK_ID_CURR", "SK_ID_PREV", "N", "V", "DPD"])
    data = data.agg({"REPAID": "sum"})
    data = data.reset_index(level=["N", "V", "DPD"])
    data = data[["N", "V", "REPAID", "DPD"]]
    return data


def get_clean_installments_payments_expected_dpd_by_loan(
    repaid_and_dpd: pd.DataFrame,
    loan_amount: pd.DataFrame
) -> pd.DataFrame:
    data = repaid_and_dpd.merge(
        loan_amount, on=["SK_ID_CURR", "SK_ID_PREV"], how="left"
    )
    data["PCT_REPAID"] = data.REPAID / data.AMT_LOAN
    data["EXP_DPD"] = data.DPD * data["PCT_REPAID"]

    data = data.groupby(by=["SK_ID_CURR", "SK_ID_PREV"]).agg({
        "REPAID": "sum",
        "AMT_LOAN": "first",
        "PCT_REPAID": "sum",
        "EXP_DPD": "sum"
    })
    
    data.columns.name = "EXP_DPD_BY_LOAN"
    return data


def get_clean_installments_payments_expected_dpd_by_client(
    repaid_and_dpd: pd.DataFrame,
    loaned_amount: pd.DataFrame
) -> pd.DataFrame:
    data = repaid_and_dpd.merge(
        loaned_amount, on="SK_ID_CURR", how="left"
    )
    data["PCT_REPAID"] = data.REPAID / data.AMT_LOANED
    data["EXP_DPD"] = data.DPD * data["PCT_REPAID"]

    data = data.groupby(by="SK_ID_CURR").agg({
        "REPAID": "sum",
        "AMT_LOANED": "first",
        "PCT_REPAID": "sum",
        "EXP_DPD": "sum"
    })

    data.columns.name = "EXP_DPD_BY_CLIENT"
    return data


def get_clean_installments_payments_base(
    data: pd.DataFrame
) -> pd.DataFrame:
    base_data = get_installments_payments_by_installment_and_version(data)
    na_inst_case = base_data.AMT_INSTALMENT == 0
    base_data.loc[na_inst_case, "AMT_INSTALMENT"] = base_data[na_inst_case].AMT_PAYMENT
    return base_data



def get_installments_payments_by_installment(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate installment payment data by installment.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing installment payment data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with aggregated installment payment information, including
        the minimum and maximum version numbers, the count of versions, the
        sum of installment amounts, and the last payment amount.

    Notes
    -----
    This function aggregates installment payment data based on the installment
    number, summarizing version-related information.

    The input `data` should be the output of the
    `get_installments_payments_by_installment_and_version`
    function.
    """
    # Reset the index for grouping
    data = data.reset_index()

    # Group the data by relevant columns
    data = data.groupby(by=[
        "SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_NUMBER"
    ])

    # Perform and return aggregation
    data = data.agg({
        "NUM_INSTALMENT_VERSION" : ["min", "max", "count"],
        "AMT_INSTALMENT" : "sum",
        "AMT_PAYMENT": "last",
        "DAYS_INSTALMENT" : ["max", "min"],
        "CNT_PAYMENT": "sum",
        "DAYS_ENTRY_PAYMENT_START": "max",
        "DAYS_ENTRY_PAYMENT_END": "min"
    })
    
    data.columns = [
        "V_MIN", "V_MAX", "V_COUNT", "AMT_INSTALMENT", "AMT_PAYMENT",
        "DAYS_INSTALMENT_START", "DAYS_INSTALMENT_END",
        "CNT_PAYMENT", "DAYS_ENTRY_PAYMENT_START", "DAYS_ENTRY_PAYMENT_END"
    ]
    
    return data



""" DEPRECATED remove after check of dependencies
"""


def get_installments_payments_by_version_deprecated(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    DEPRECATED  False hypothesis
    Use get_installments_payments_by_installment instead
    
    Aggregate installment payment data by version.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing installment payment data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with aggregated installment payment information, including
        the minimum and maximum version numbers, the count of versions, the
        minimum and maximum installment amounts, and the minimum and maximum
        payment amounts.

    Notes
    -----
    This function aggregates installment payment data based on the installment
    number, summarizing version-related information.

    The input `data` should be the output of the
    `get_installments_payments_by_installment_and_version`
    function.
    """
    # Reset the index for grouping
    data = data.reset_index()

    # Group the data by relevant columns
    data = data.groupby(by=[
        "SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_NUMBER"
    ])

    # Perform and return aggregation
    return data.agg({
        "NUM_INSTALMENT_VERSION" : ["min", "max", "count"],
        "AMT_INSTALMENT" : ["min", "max"],
        "AMT_PAYMENT": ["min", "max"]
    })


def filter_installments_payments_by_version_outliers_deprecated(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    DEPRECATED False hypothesis

    Filter installment payment data by version outliers.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing installment payment data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with filtered installment payment data, excluding rows
        where the maximum payment amount is less than or equal to the minimum
        payment amount.

    Notes
    -----
    This function filters installment payment data to remove rows where the
    maximum payment amount is less than or equal to the minimum payment amount.

    The input `data` should be the output of the `get_installments_payments_by_version`
    function.
    """
    pyt_max = data[("AMT_PAYMENT", "max")]
    pyt_min = data[("AMT_PAYMENT", "min")]

    return data[pyt_max > pyt_min]


def _get_clean_installments_payments_by_installment_and_version_deprecated(
    data: pd.DataFrame,
    outliers: pd.DataFrame
) -> pd.DataFrame:
    """
    DEPRECATED False hypothesis
    
    Get clean installment payment data by installment and version.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing installment payment data.

    outliers : pd.DataFrame
        The DataFrame containing version outliers identified for filtering.

    Returns
    -------
    pd.DataFrame
        A DataFrame with cleaned installment payment data after removing rows
        associated with version outliers.

    Notes
    -----
    This function filters installment payment data based on version outliers
    provided in the `outliers` DataFrame.
    """
    # Reset the index for grouping
    drop_index = data.reset_index(level=3)
    
    # Extract the NUM_INSTALMENT_VERSION column
    drop_index = drop_index[["NUM_INSTALMENT_VERSION"]]
    
    # Locate rows with outliers
    drop_index = drop_index.loc[outliers.index]
    
    # Reset the index
    drop_index = drop_index.reset_index()
    
    # Identify rows to keep (non-duplicates)
    keep_index = drop_index.drop_duplicates(
        subset=drop_index.columns[:-1],
        keep="last"
    )
    
    # Filter out rows based on the keep_index, keeping only non-duplicate rows
    drop_index = drop_index[~drop_index.index.isin(keep_index.index)]
    
    # Create a MultiIndex from the rows to drop
    drop_index = pd.MultiIndex.from_frame(drop_index)
    
    # Create a mask to filter the data
    mask = ~data.index.isin(drop_index)
    
    return data[mask]


def get_clean_installments_payments_by_installment_and_version_deprecated(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    DEPRECATED False hypothesis
    
    Get clean installment payment data by installment and version.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing installment payment data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with cleaned installment payment data after removing rows
        associated with version outliers.
    """
    aggregated_by_version = get_installments_payments_by_installment_and_version(data)
    aggregated_version = get_installments_payments_by_version_deprecated(aggregated_by_version)
    outliers = filter_installments_payments_by_version_outliers_deprecated(aggregated_version)
    return _get_clean_installments_payments_by_installment_and_version_deprecated(
        aggregated_by_version,
        outliers
    ).copy()



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
