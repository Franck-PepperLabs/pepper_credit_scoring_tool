"""Module: pepper/monovar.py

This module provides a collection of functions for analyzing and reporting
data, including functions to compute statistics, visualize data, and export
reports to Google Sheets. It also includes functions for calculating
correlation metrics and working with class weights for machine learning tasks.

Functions:
- `series_infos(s: pd.Series, idx: int) -> Dict[str, Union[str, int, float, bool, List[float], List[str]]]`:
    Analyze a Pandas Series and return information about it.

- `dataframe_infos(df: pd.DataFrame) -> pd.DataFrame`:
    Analyze a Pandas DataFrame and return information about all its Series.

- `data_report(data: Union[pd.DataFrame, pd.Series], max_mods: int = 30) -> pd.DataFrame`:
    Generate a data report, perform data reduction, and return it as a DataFrame.

- `data_report_to_csv_file(report: pd.DataFrame, csv_filename: str) -> None`:
    Export a data report to a CSV file.

- `data_report_to_gsheet(report: pd.DataFrame, spread, sheet_name: str) -> None`:
    Export a data report to a Google Sheets spreadsheet.

- `agg_value_counts(s: pd.Series, agg: Union[None, bool, float, int] = .01, dropna: bool = True) -> pd.DataFrame`:
    Compute value counts and relative frequencies of a Pandas Series.

- `target_weights(target: np.ndarray) -> Tuple[float, float]`:
    Calculate the weights for each class in a binary target array.

- `get_sample_weights(target: np.ndarray) -> np.ndarray`:
    Compute sample weights for each element in a binary target array.

- `wmcc(target: np.ndarray, var: np.ndarray) -> float`:
    Compute the weighted Matthews correlation coefficient between two arrays.

- `weighted_kendall_tau(target: np.ndarray, var: np.ndarray) -> float`:
    Calculate the weighted Kendall's Tau rank correlation between two arrays.

- `show_correlations(data: pd.DataFrame, title: str, method: str = "pearson", ratio: float = 1) -> pd.DataFrame`:
    Compute pairwise correlations between columns in a Pandas DataFrame and
    display the correlation matrix as a heatmap.

- `test_wmcc(target) -> None`:
    Test the `wmcc` function using various scenarios.

This module is a comprehensive toolkit for data analysis, reporting, and
machine learning preparation, ensuring that project data is effectively
processed and analyzed.
"""

from typing import Union, Tuple, List, Dict

import os
import pandas as pd
import numpy as np

import scipy.stats as sps
from statsmodels import robust
from sklearn.metrics import matthews_corrcoef

import matplotlib.pyplot as plt
import seaborn as sns

from pepper.env import get_tmp_dir
from pepper.utils import set_plot_title, save_and_show
from pepper.gsheet_io import df_to_gsheet

from IPython.display import display
from gspread_pandas import Spread

import pandas as pd
import numpy as np
from typing import Union, List, Dict
from scipy import stats as sps


def _is_numeric(s: pd.Series) -> bool:
    """
    Check if a Series is numeric.

    Parameters
    ----------
    s : pd.Series
        The input Series.

    Returns
    -------
    bool
        True if the Series is numeric, False otherwise.
    """
    return pd.api.types.is_numeric_dtype(s.dtype)


def _get_counts_and_freqs(s: pd.Series) -> (pd.Series, pd.Series):
    """
    Calculate value counts and frequencies for a Series.

    Parameters
    ----------
    s : pd.Series
        The input Series.

    Returns
    -------
    (pd.Series, pd.Series)
        A tuple containing two Series:
        the value counts and the normalized frequencies.
    """
    counts = s.value_counts()
    freqs = s.value_counts(normalize=True)
    return counts, freqs


def _get_n_na_and_n_notna(s: pd.Series) -> (int, int):
    """
    Calculate the number of missing and non-missing values in a Series.

    Parameters
    ----------
    s : pd.Series
        The input Series.

    Returns
    -------
    (int, int)
        A tuple containing two integers:
        the number of missing values and the number of non-missing values.
    """
    n_na = s.isna().sum()
    n_notna = s.notna().sum()
    return n_na, n_notna


def _get_type_info(s: pd.Series) -> Dict:
    """
    Get type-related information for a Series.

    Parameters
    ----------
    s : pd.Series
        The input Series.

    Returns
    -------
    Dict
        A dictionary containing information about the Series' data type
        and related properties.
    """
    is_numeric = _is_numeric(s)
    return {
        # Identification, group, and type
        'group': '<NYI>',  # TODO: Implement group(idx)
        'subgroup': '<NYI>',  # TODO: Implement subgroup(idx)
        'name': s.name,
        'domain': '<NYI>',  # TODO: Implement domain(idx)
        'format': '<NYI>',
        'dtype': s.dtype,
        'astype': '<NYI>',
        'unity': '<NYI>',
        'is_numeric': is_numeric
    }


def _get_counts_info(s: pd.Series):
    """
    Calculate counts and statistics for a Series.

    Parameters
    ----------
    s : pd.Series
        The input Series.

    Returns
    -------
    Dict
        A dictionary containing various statistics about the Series,
        including the number of missing values, the number of unique values,
        the filling rate, and uniqueness (if applicable).
    """
    # Number of missing values, unique values, and filling rate
    precision = 3
    n = s.size
    n_na, n_notna = _get_n_na_and_n_notna(s)
    n_unique = s.nunique()
    return {
        'n': n,
        'hasnans': s.hasnans,
        'n_unique': n_unique,
        'n_notna': n_notna,
        'n_na': n_na,
        'filling_rate': round(n_notna / n, precision),
        'uniqueness': round(n_unique / n_notna, precision) if n_notna else pd.NA
    }


def _get_numvar_info(s: pd.Series) -> Dict:
    """
    Calculate statistics for a numeric Series.

    Parameters
    ----------
    s : pd.Series
        The input Series.

    Returns
    -------
    Dict
        A dictionary containing various statistics for the numeric Series,
        including minimum and maximum values, mode (up to 10 most common values),
        mean, trimmed mean, median, standard deviation, interquartile range,
        median absolute deviation, skewness, kurtosis, and value interval.
    """
    # Distribution for numeric variable
    if not _is_numeric(s):
        return {
            'val_min': None, 'val_max': None, 'val_mode': None, 'val_mean': None,
            'val_trim_mean_10pc': None, 'val_med': None, 'val_std': None,
            'val_interq_range': None, 'val_med_abs_dev': None,
            'val_skew': None, 'val_kurt': None, 'interval': None
        }
    s = s.astype(float)
    return {
        # Central tendency
        'val_min': s.min(),
        'val_max': s.max(),
        'val_mode': str([round(m, 2) for m in s.mode().tolist()[:10]]),
        'val_mean': round(s.mean(), 3),
        'val_trim_mean_10pc': round(sps.trim_mean(s.dropna().values, 0.1), 3),
        'val_med': s.median(),
        # Dispersion
        'val_std': round(s.std(), 3),
        'val_interq_range': s.quantile(0.75) - s.quantile(0.25),
        'val_med_abs_dev': robust.mad(s.dropna()),
        'val_skew': round(s.skew(), 3),
        'val_kurt': round(s.kurtosis(), 3),
        'interval': [s.min(), s.max()]
    }


def _get_catvar_info(s: pd.Series) -> Dict:
    """
    Calculate statistics for a categorical Series.

    Parameters
    ----------
    s : pd.Series
        The input Series.

    Returns
    -------
    Dict
        A dictionary containing statistics for the categorical Series,
        including modalities (unique values), mod_counts (counts for each modality),
        and mod_freqs (frequencies for each modality).
    """
    # Distribution for categorical variable
    if _is_numeric(s):
        return {'modalities': None, 'mod_counts': None, 'mod_freqs': None}
    counts, freqs = _get_counts_and_freqs(s)
    return {
        'modalities': list(counts.index),
        'mod_counts': list(counts),
        'mod_freqs': list(freqs),
    }


def _get_technical_info(s: pd.Series) -> Dict:
    """
    Calculate statistics for a categorical Series.

    Parameters
    ----------
    s : pd.Series
        The input Series.

    Returns
    -------
    Dict
        A dictionary containing technical information about the Series,
        including its shape, dimensionality, emptiness, size, number of bytes
        consumed by the Series, memory usage, flags, array container type,
        and values container type.
    """
    # Dimensions, memory footprint, implementation metadata
    return {
        'shape': s.shape,
        'ndim': s.ndim,
        'empty': s.empty,
        'size': s.size,
        'nbytes': s.nbytes,
        'memory_usage': s.memory_usage,     # not reported in the data dict
        'flags': s.flags,
        'array container type': type(s.array),
        'values container type': type(s.values)
    }


def series_infos(
    s: Union[pd.Series, np.ndarray, List],
    idx: int = 0
) -> Dict[str, Union[str, int, float, bool, List[float], List[str]]]:
    """
    Analyze a Series and return information about it.

    Parameters
    ----------
    s : Union[pd.Series, np.ndarray, List]
        The Series or array to analyze.
    idx : int
        Index of the Series. Defaults to 0.

    Returns
    -------
    Dict[str, Union[str, int, float, bool, List[float], List[str]]]
        Information about the Series.

    Notes
    -----
    - 'group', 'subgroup', 'domain', 'format', 'astype', 'unity' fields are not yet implemented.
    - 'val_mode' field contains a list of up to 10 most common values, rounded to 2 decimal places.
    - 'val_trim_mean_10pc' field calculates the trimmed mean with 10% of values removed.
    - 'interval' field contains the minimum and maximum values of the numeric Series.
    - 'modalities', 'mod_counts', 'mod_freqs' fields are provided for non-numeric Series.
    - 'memory_usage', 'flags', 'array container type', and 'values container type' fields are metadata about the Series implementation.
    """
    if isinstance(s, (np.ndarray, list)):
        s = pd.Series(s)

    info = {'idx': idx}
    info |= _get_type_info(s)
    info |= _get_counts_info(s)
    info |= _get_numvar_info(s)
    info |= _get_catvar_info(s)
    info |= _get_technical_info(s)

    return info


def dataframe_infos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze a DataFrame and return information about all its Series.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.

    Returns
    -------
    pd.DataFrame
        Information about all the Series in the DataFrame.
    """
    infos = [
        pd.Series(series_infos(df[c], df.columns.get_loc(c)))
        for c in df.columns
    ]
    return pd.DataFrame(infos)


def data_report(
    data: Union[pd.DataFrame, pd.Series],
    max_mods: int = 30
) -> pd.DataFrame:
    """
    Generate a data report.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        The data to analyze. Can be either a DataFrame or a Series.
    max_mods : int, optional
        The maximum number of modalities to report. Defaults to 30.

    Returns
    -------
    pd.DataFrame
        Information about the data.

    Notes
    -----
    This function generates a report on the provided data, including various statistics and information.
    """
    def cut_cat_list(x):
        if x is None:
            return None
        if len(x) > max_mods:
            return x[:max_mods] + ["..."]

    def cut_num_list(x):
        if x is None:
            return None
        if len(x) > max_mods:
            return x[:max_mods] + [sum(x[30:])]
    
    # build the dataframe version
    report = dataframe_infos(data)
    
    if max_mods is not None:
        # some reductions before export
        report.modalities = report.modalities.apply(cut_cat_list)
        report.mod_counts = report.mod_counts.apply(cut_num_list)
        report.mod_freqs = report.mod_freqs.apply(cut_num_list)

    report.drop(columns=["memory_usage"], inplace=True)

    return report


def data_report_to_csv_file(
    report: pd.DataFrame,
    csv_filename: str
) -> None:
    """
    Export a data report to a CSV file.

    Parameters
    ----------
    report : pd.DataFrame
        The data report.
    csv_filename : str
        The filename for the CSV report.

    Returns
    -------
    None
    """
    # save in CSV
    tmp_dir = get_tmp_dir()
    vars_analysis_path = os.path.join(tmp_dir, csv_filename)
    report.to_csv(vars_analysis_path, sep=",", encoding="utf-8", index=False)


def data_report_to_gsheet(
    report: pd.DataFrame,
    spread: Spread,
    sheet_name: str
) -> None:
    """
    Export data report to a Google Sheets spreadsheet.

    Parameters
    ----------
    report : pd.DataFrame
        The data report as a DataFrame.
    spread : gspread_pandas.Spread
        The Google Sheets spreadsheet.
    sheet_name : str
        The name of the sheet in the Google Sheets spreadsheet.

    Returns
    -------
    None

    Notes
    -----
    This function exports the data report to a Google Sheets spreadsheet.
    It performs data conversion and formatting suitable for Google Sheets.
    
    Example
    -------
    from pepper.univar import data_report, data_report_to_gsheet
    from gspread_pandas import Spread
    # target GSheet
    spread = Spread("<Your Google Doc ID>")
    # build the report dataframe
    report = data_report(data)
    # export to the 'data_dict' sheet
    data_report_to_gsheet(report, spread, 'data_dict')
    """
    numeric_slice = slice("filling_rate", "mod_freqs")
    numeric_cols = list(report.loc[:, numeric_slice].columns)
    df_to_gsheet(
        data=report,
        spread=spread,
        sheet_name=sheet_name,
        as_code=["shape"],
        as_fr_FR=numeric_cols,
        start="A7"
    )


""" Cats analysis
"""


def print_value_counts_dict(data: pd.DataFrame, col: str, normalize=False):
    vc_dict = data[col].value_counts(normalize=normalize, dropna=False).to_dict()
    print(f"{col} ({len(vc_dict)}): {vc_dict}")


def agg_value_counts(
    s: pd.Series,
    agg: Union[None, bool, float, int] = .01,
    dropna: bool = True
) -> pd.DataFrame:
    """
    Compute the value counts and relative frequencies of a pandas Series.
    
    Parameters
    ----------
    s : pd.Series
        The pandas Series.
    agg : Union[None, bool, float, int], optional
        An optional parameter that specifies how to aggregate the results.
        If None, returns the raw value counts and relative frequencies.
        If a float between 0 and 1, returns a DataFrame with a row containing
        all values whose cumulative proportion of occurrences is less than agg,
        with the remaining values aggregated. If True, determines the threshold
        automatically using the first index at which the proportion is less
        than the sum of the next proportions.
    dropna : bool, default True
        Don't include counts of NaN.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with two columns: 'count', which contains the
        absolute frequency of each value, and 'proportion', which contains the
        relative frequency of each value. If aggregation is used, the last row
        of the DataFrame will show the aggregated counts and proportions.
    
    Examples
    --------
    >>> s = app_amt_req.AMT_REQ_CREDIT_BUREAU_MON.dropna().astype(int)
    >>> display(agg_value_counts(s))
    >>> display(agg_value_counts(s, agg=True))
    >>> display(agg_value_counts(s, agg=.15))
    
    TODO Refactor
    """
    # Compute the absolute and relative value counts of the series
    abs_vc = s.value_counts(dropna=dropna)
    rel_vc = s.value_counts(normalize=True, dropna=dropna)

    # Combine the absolute and relative value counts into a single DataFrame
    vc = pd.concat([abs_vc, rel_vc], axis=1)
    
    if agg is None or s.dtype == bool:
        # If no aggregation is requested,
        # return the raw value counts and relative frequencies
        return vc
    else:
        # Compute the cumulative sum of the relative freqs in reverse order
        rev_rel_vc = rel_vc.loc[rel_vc.index[::-1]]
        rel_vc_rcumsum = rev_rel_vc.cumsum().loc[rel_vc.index]

        # Determine which values to aggregate
        if isinstance(agg, (int, float)) and not isinstance(agg, bool):
            # If a threshold value is specified, identify the values whose
            # cumulative proportion is less than the threshold
            is_epsilon = rel_vc_rcumsum < agg
        elif isinstance(agg, bool) and agg:
            # If automatic thresholding is requested, identify the first index
            # at which the proportion is less than the sum of the next ones
            """agg_start_idx = np.argmin(rel_vc > rel_vc_rcumsum.shift(-1))
            is_epsilon = vc.index >= agg_start_idx"""
            # Avoid a one bar graph :
            is_sup_to_remainder = rel_vc > rel_vc_rcumsum.shift(-1)
            first_true_idx = np.argmax(is_sup_to_remainder)
            agg_start_idx = first_true_idx + np.argmin(is_sup_to_remainder[first_true_idx:])
            is_epsilon = np.array(
                [False] * agg_start_idx
                + [True] * (vc.shape[0] - agg_start_idx)
            )
            # is_epsilon = vc.index >= agg_start_idx

        # Avoid a one class aggregator
        if is_epsilon.sum() < 2:
            return vc

        # Aggregate the values as necessary
        agg_vc = vc[~is_epsilon].copy()
        eps_vc = vc[is_epsilon].copy()
        eps_vc.sort_index(inplace=True)
        eps_idx = f"{eps_vc.index[0]}:{eps_vc.index[-1]}"
        
        # Don't do this (undesired float cast that must then be fixed by another cast) :
        # agg_vc.loc[eps_idx] = eps_vc.sum()
        # agg_vc.astype({"count": int}, copy=False)
        # the right solution is :
        sums_row = pd.DataFrame(
            {"count": eps_vc["count"].sum(), "proportion": eps_vc["proportion"].sum()},
            index=[eps_idx]
        )
        agg_vc = pd.concat([agg_vc, sums_row])
        agg_vc.index.name = s.name
        return agg_vc


""" P4 : totalement dépassé
def cats_freqs(data, label):
    c = data[label]
    f = c.value_counts()
    f.plot.bar()   # TODO : faire plus sympa à l'aide de SNS
    plt.show()
    return f
"""

# P4 à refaire en mieux
def cats_weighted_freqs(
    data: pd.DataFrame,
    label: str,
    wlabels: list,
    short_wlabels: list
) -> pd.DataFrame:
    """
    Compute weighted frequencies of categorical values in a DataFrame column.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data.
    label : str
        The name of the column for which to calculate frequencies.
    wlabels : list
        A list of column names containing weights.
    short_wlabels : list
        A list of short names for weight columns for plotting.

    Returns
    -------
    pd.DataFrame
        A DataFrame with weighted frequencies of categorical values.

    Examples
    --------
    >>> fw = cats_weighted_freqs(
            data, 'label_column',
            ['weight_column1', 'weight_column2'],
            ['W1', 'W2']
        )
    >>> display(fw)
    """
    cw = data[[label] + wlabels]
    aggs = {label: "count"}
    for wlabel in wlabels:
        aggs[wlabel] = "sum"
    fw = cw.groupby(by=label).agg(aggs)
    fw.columns = ["count"] + short_wlabels
    fw = fw.sort_values(by=fw.columns[1], ascending=False)
    total = fw.sum()
    fw /= total
    # TODO : faire plus sympa à l'aide de SNS
    fw.plot.bar(figsize=(.5 * fw.shape[0], 5))
    plt.show()
    return fw


""" Correlation
"""

def target_weights(target: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the weights of each class in the target array.
    
    Parameters
    ----------
    target : np.ndarray
        The target variable as a binary array (0s and 1s).
        
    Returns
    -------
    Tuple[float, float]
        A tuple containing the weights for class 0 and class 1, respectively.
        
    Example
    -------
    >>> target = np.array([0, 1, 1, 0, 1])
    >>> target_weights(target)
    (1.25, 5.0)
    """
    n = len(target)
    n_1 = (target == 1).sum()
    n_0 = n - n_1
    w_0 = n / (1 + n_0)  # 1 + n to avoid div by 0
    w_1 = n / (1 + n_1)
    return w_0, w_1


def get_sample_weights(target: np.ndarray) -> np.ndarray:
    """
    Compute the sample weights for each element in the target array.
    
    Parameters
    ----------
    target : np.ndarray
        The target variable as a binary array (0s and 1s).
        
    Returns
    -------
    np.ndarray
        An array with the sample weights for each element in the target array.
        
    Example
    -------
    >>> target = np.array([0, 1, 1, 0, 1])
    get_sample_weights(target)
    array([1.25, 5.  , 5.  , 1.25, 5.  ])
    """
    w_0, w_1 = target_weights(target)
    # Build a ndarray with weight for each sample
    return target * w_1 + (1 - target) * w_0


def wmcc(target: np.ndarray, var: np.ndarray) -> float:
    """
    Compute the weighted Matthews correlation coefficient between two arrays.

    Parameters
    ----------
    target : np.ndarray
        The target variable as a binary array (0s and 1s).
    var : np.ndarray
        The variable to compare to the target variable, as an array of any type
        with an order relation.

    Returns:
    --------
    float
        The weighted Matthews correlation coefficient between the two arrays.

    Example:
    --------
    >>> target = np.array([0, 0, 1, 1, 1])
    >>> var = np.array([1, 0, 1, 0, 1])
    >>> wmcc(target, var)
    0.25819888974716115
    """
    return matthews_corrcoef(
        target, var,
        sample_weight=get_sample_weights(target)
    )


from scipy import stats
def weighted_kendall_tau(target: np.ndarray, var: np.ndarray) -> float:
    """
    Calculate the weighted Kendall's Tau rank correlation between two arrays.
    
    Parameters
    ----------
    target : np.ndarray
        The target variable as a binary array (0s and 1s).
    var : np.ndarray
        The variable to compare to the target variable, as an array of any type
        with an order relation.
        
    Returns
    -------
    float
        The weighted Kendall's Tau rank correlation between the two arrays.
        
    Example
    -------
    # generate sample data
    >>> target = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    >>> var = np.array([1.2, 2.5, 0.8, 1.5, 1.7, 0.9, 2.1, 1.1, 1.8, 0.7])
    # calculate weighted Kendall's Tau rank correlation
    >>> weighted_kendall_tau(target, var) 
    0.13636363636363635
    
    
    Description
    -----------
    Kendall's Tau rank correlation measures the association between two arrays by comparing the
    number of concordant and discordant pairs of values. The weighted version of Kendall's Tau
    assigns different weights to different pairs, making it suitable for data with varying degrees
    of importance.
    """
    w_0, w_1 = target_weights(target)
    corr, p_value = stats.weightedtau(
        target, var,
        weigher=lambda x: w_1 if x else w_0
    )
    # print("Weighted Kendall's Tau Rank Correlation : ", corr)
    # print("P-value : ", p_value)
    return corr  


def show_correlations(
    data: pd.DataFrame,
    title: str,
    method: str = "pearson",
    ratio: float = 1
) -> pd.DataFrame:
    """
    Compute pairwise correlation of columns in the input dataframe using
    the specified correlation method and plot the correlation matrix as a
    heatmap.

    Parameters
    ----------
    data : pd.DataFrame
        The input data to compute the correlation matrix on.
    title : str
        The title of the plot.
    method : str, optional
        The correlation method to use, can be one of {'pearson', 'spearman',
        'kendall', 'matthews'}, by default 'pearson'.
    ratio : float, optional
        The aspect ratio of the plot, by default 1.

    Returns
    -------
    pd.DataFrame
        The computed pairwise correlation of columns in the input dataframe.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame({
            "col_1": np.random.normal(0, 1, 100),
            "col_2": np.random.normal(0, 1, 100),
            "target": np.random.binomial(1, 0.5, 100)
        })
    >>> corr_matrix = show_correlations(
    >>>     data, "Correlation matrix", method="spearman"
    >>> )
    """
    data_corr = None
    if method == "matthews":
        data_corr = data.corr(method=wmcc)
    elif method == "kendall":
        data_corr = data.corr(method=weighted_kendall_tau)
    else:
        data_corr = data.corr(method=method)
    
    # Draw a heatmap with the numeric values in each cell
    _, ax = plt.subplots(figsize=(8 * ratio, 6 * ratio))
    sns.heatmap(data_corr, annot=True, ax=ax)
    title = f"{title} ({method})"
    set_plot_title(title)
    plt.ylabel("")
    plt.xlabel("")
    plt.xticks(rotation=30, ha="right", rotation_mode="anchor")

    # Save and show the plot
    save_and_show(f"{title.lower()}", sub_dir="corr")
    return data_corr


def test_wmcc(target: np.ndarray) -> None:
    """
    Test the weighted Matthews correlation coefficient (WMCC) function with various inputs.

    Parameters
    ----------
    target : np.ndarray
        The target variable as a binary array (0s and 1s).

    Returns
    -------
    None

    Example
    -------
    >>> target = np.array([0, 0, 1, 1, 1])
    >>> test_wmcc(target)
    
    Description
    -----------
    This function tests the `wmcc` function, which calculates the
    weighted Matthews correlation coefficient (WMCC) between two arrays.
    It checks the WMCC with various inputs, including:
    - The WMCC of the target array with itself.
    - The WMCC of the target array with its complement (1s flipped to 0s and vice versa).
    - Checking if shuffling the target array affects the WMCC.
    
    The function displays the results of these tests for verification.
    """
    display(wmcc(target, target))
    display(wmcc(target, 1 - target))
    m = len(target) // 3
    shuffled_target = target.values.copy()
    np.random.shuffle(shuffled_target[m:2*m])
    display((target == shuffled_target).all())
    display(wmcc(
        target,
        shuffled_target
    ))
