from typing import Any, List, Tuple, Iterator, Optional
import os
from datetime import datetime, time

import pandas as pd


from sklearn import ensemble, preprocessing

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    TargetDriftPreset,
    # RegressionMetric,
    RegressionPreset
)

from home_credit.load import get_reports_dir
from home_credit.best_model_search import post_train_eval


def refine_index(data: pd.DataFrame) -> None:
    """Refines the index of a DataFrame by combining the date column and the
    hour column to create a new datetime index.
    
    Parameters
    ----------
    data : pd.DataFrame
        The data with separate date and hour columns.
        
    Returns
    -------
    None
        The index of `data` is modified in place.
    """
    # TODO, il me semble que l'on peut faire mieux
    # Ici, un df.apply par rows n'est pas forcément idéal
    def hr_suffix(row: pd.Series) -> Any:
        return datetime.combine(row.name, time(hour=int(row.hr)))
    
    data.index = data.apply(hr_suffix, axis=1)


def split_feats(
    data: pd.DataFrame,
    nu_thres: int
) -> Tuple[List[str], List[str], List[str]]:
    """Splits the features in a dataframe into three categories: primary keys,
    categorical and numerical based on the number of unique values in each
    column.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the features to split.
    nu_thres : int
        The threshold for the number of unique values to use to distinguish
        between categorical and numerical features.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        A tuple containing three lists: primary keys, categorical features,
        numerical features.
    """
    # We can exclude variables with a Shannon entropy of 1 : they are pks
    # categorical iff <= nu_thresh else numerical
    n_samples = data.shape[0]
    nu = data.nunique()
    pks = nu[nu == n_samples].index.to_list()
    cat_feats = nu[nu <= nu_thres].index.to_list()
    num_feats = nu[(nu_thres < nu) & (nu < n_samples)].index.to_list()
    return pks, cat_feats, num_feats


def datetime_range(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    time_step: str
) -> Iterator[pd.DataFrame]:
    """Slices a time range into regularly spaced intervals.
    
    Parameters
    ----------
    start_date : pd.Timestamp
        The start date of the time range.
    end_date : pd.Timestamp
        The end date of the time range.
    time_step : str
        The time step to use for the intervals,
        e.g. '1D' for one day or '7D' for one week.
        
    Yields
    ------
    pd.DataFrame
        A slice of the time range as a Pandas DataFrame.
    """
    freq = pd.date_range(start_date, end_date, freq=time_step)
    for i in range(len(freq)-1):
        yield (freq[i], freq[i+1])


def get_col_map(
    y_true: pd.DataFrame, 
    y_pred: pd.DataFrame, 
    num_feats: List[str], 
    cat_feats: List[str]
) -> ColumnMapping:
    """Create a mapping between columns of true and predicted dataframes.

    Parameters
    ----------
    y_true : pd.DataFrame
        The true values dataframe.
    y_pred : pd.DataFrame
        The predicted values dataframe.
    num_feats : List[str]
        The list of numerical feature names.
    cat_feats : List[str]
        The list of categorical feature names.

    Returns
    -------
    ColumnMapping
        The mapping between true and predicted dataframes.
    """
    return ColumnMapping(
        target=y_true, 
        prediction=y_pred,
        numerical_features=num_feats,
        categorical_features=cat_feats
    )


def report_performance(
    data_ref: pd.DataFrame, 
    data_curr: pd.DataFrame,
    col_map: ColumnMapping,
    metrics: Optional[List] = [RegressionPreset()]
) -> Report:
    """Reports the performance of a model by comparing the reference data and
    the current data.
    
    Parameters
    ----------
    data_ref : pd.DataFrame
        The reference data.
    data_curr : pd.DataFrame
        The current data.
    col_map : ColumnMapping
        The column mapping between the reference and current data.
    metrics : Optional[List], optional
        The metrics to use to report performance,
        by default [RegressionPreset()].
    
    Returns
    -------
    Report
        The report object containing the performance metrics.
    """
    report = Report(metrics=metrics)
    report.run(
        reference_data=data_ref,
        current_data=data_curr,
        column_mapping=col_map
    )
    return report


def report_datadrift(
    data_ref: pd.DataFrame, 
    data_curr: pd.DataFrame, 
    col_map: ColumnMapping,
    metrics: Optional[List] = [DataDriftPreset()]
) -> Report:
    """Reports the dat drift of a model by comparing the reference data and
    the current data.
    
    Parameters
    ----------
    data_ref : pd.DataFrame
        The reference data.
    data_curr : pd.DataFrame
        The current data.
    col_map : ColumnMapping
        The column mapping between the reference and current data.
    metrics : Optional[List], optional
        The metrics to use to report performance,
        by default [RegressionPreset()].
    
    Returns
    -------
    Report
        The report object containing the data drift metrics.
    """
    return report_performance(data_ref, data_curr, col_map, metrics)


def save_report(report: Report, report_name: str) -> None:
    """
    Saves a report to an HTML file.
    
    Parameters
    ----------
    report : Report
        The report object to save.
    report_name : str
        The name of the report file.
    """
    report.save_html(
        os.path.join(
            get_reports_dir(),
            f"{report_name}.html"
    ))

