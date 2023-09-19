from typing import Union, Optional

from IPython.display import display

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
pd.options.display.float_format = "{:.3f}".format

import matplotlib.pyplot as plt
import seaborn as sns

from pepper.utils import print_subtitle, save_and_show
from pepper.univar import agg_value_counts
from pepper.feat_eng import reduce_long_tail
from pepper.plots import show_cat_mod_counts  #, lin_log_tetra_histplot

from home_credit.load import get_table  # get_columns_description, 
from home_credit.merge import targetize
from home_credit.utils import help_cols


""" Data blocks NB macros
"""


def get_datablock(data: Union[pd.DataFrame, str], pat: str) -> pd.DataFrame:
    """
    Extract a subset of columns from a DataFrame based on a regular expression pattern.

    Parameters:
    -----------
    data : Union[pd.DataFrame, str]
        The input DataFrame or the name of the table to load and use.
    pat : str
        The regular expression pattern to match column names.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing only the columns that match the pattern.

    Examples:
    ---------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> get_datablock(df, 'A')
       A
    0  1
    1  2
    2  3

    >>> get_datablock('my_table', 'col_\\d+')
    # Loads the 'my_table' table, extracts columns with names matching the pattern 'col_\d+'.
    """
    if isinstance(data, str):
        data = get_table(data).copy()
    return data[data.columns[data.columns.str.match(pat)]]


def get_labeled_datablock(table_name: str, pat: str) -> pd.DataFrame:
    """
    Extract a labeled subset of columns from a DataFrame based on a regular expression pattern.

    If the DataFrame does not contain a 'TARGET' column, it will create one by calling 'targetize' on the 'SK_ID_CURR' column.

    Parameters:
    -----------
    table_name : str
        The name of the table to load and use.
    pat : str
        The regular expression pattern to match column names.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing columns matching the pattern 'pat' and the 'TARGET' column.

    Examples:
    ---------
    >>> get_labeled_datablock('my_table', 'col_\\d+')
    # Loads the 'my_table' table, extracts columns with names matching the pattern 'col_\d+' and includes the 'TARGET' column.
    """
    data = get_table(table_name).copy()
    if "TARGET" not in data.columns:
        target = targetize(data.SK_ID_CURR)
        data.insert(0, "TARGET", target)
    return get_datablock(data, f"{pat}|TARGET")


def var_catdist_report(s: pd.Series, agg: bool = True) -> None:
    """
    Generate a categorical distribution report for a Series.

    This function prints a subtitle with the Series name,
    displays help information for the Series using 'help_cols',
    aggregates value counts using 'agg_value_counts',
    reduces long tail categories using 'reduce_long_tail',
    and displays the modified distribution using 'show_cat_mod_counts'.

    Parameters:
    -----------
    s : pd.Series
        The Series to generate the report for.
    agg : bool, optional
        Whether to aggregate the value counts. Default is True.

    Returns:
    --------
    None

    Examples:
    ---------
    >>> var_catdist_report(my_series, agg=True)
    # Generates a categorical distribution report for 'my_series' with aggregation.
    """
    print_subtitle(s.name)
    help_cols(s.name)
    avc = agg_value_counts(s, agg=agg)
    display(avc)
    r = reduce_long_tail(s, agg=agg)
    show_cat_mod_counts(r, order=avc.index) 


def datablock_catdist_report(data: pd.DataFrame, agg: bool = True) -> None:
    """
    Generate categorical distribution reports for columns in a DataFrame.

    This function iterates over the columns of the input DataFrame and
    generates a categorical distribution report for each column using
    the 'var_catdist_report' function.

    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame containing categorical columns to generate reports for.
    agg : bool, optional
        Whether to aggregate the value counts. Default is True.

    Returns:
    --------
    None

    Examples:
    ---------
    >>> datablock_catdist_report(my_data_frame, agg=True)
    # Generates categorical distribution reports for columns in 'my_data_frame' with aggregation.
    """
    for cat in data.columns:
        var_catdist_report(data[cat], agg=agg)


def expl_analysis_series(s: pd.Series) -> pd.Series:
    """
    Perform exploratory analysis on a pandas Series.

    This function computes various statistical measures for a given pandas Series, including minimum, maximum, mean,
    median, mode, variance, standard deviation, skewness, and kurtosis.

    Parameters:
    -----------
    s : pd.Series
        The pandas Series to analyze.

    Returns:
    --------
    pd.Series
        A Series containing the computed statistical measures.

    Examples:
    ---------
    >>> s = pd.Series([1, 2, 2, 3, 3, 3])
    >>> result = expl_analysis_series(s)
    >>> print(result)
    # Output:
    # min      1.000
    # max      3.000
    # mean     2.167
    # med      2.500
    # mod    (3.0,)
    # var      0.333
    # std      0.577
    # skew     0.276
    # kurt    -1.500
    """
    return pd.Series({
        "min": s.min(),
        "max": s.max(),
        "mean": s.mean(),
        "med": s.median(),
        "mod": tuple(s.mode()),
        "var": s.var(),
        "std": s.std(),
        "skew": s.skew(),
        "kurt": s.kurt(),
    })


def expl_analysis_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform exploratory analysis on a pandas DataFrame.

    This function applies the `expl_analysis_series` function to each column of the given DataFrame and
    returns a new DataFrame containing the computed statistical measures for each column.

    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame to analyze.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the computed statistical measures for each column.

    Examples:
    ---------
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 3, 3, 4, 5]})
    >>> result = expl_analysis_df(df)
    >>> print(result)
    # Output:
    #        A      B
    # min 1.000  2.000
    # max 5.000  5.000
    # mean 3.000  3.400
    # med  3.000  3.000
    # mod  (3.0,)  (3.0,)
    # var  2.500  1.300
    # std  1.581  1.140
    # skew 0.000  0.276
    # kurt -1.300 -1.710
    """
    return df.apply(expl_analysis_series)


""" Credit default rate by a var classes
"""


def get_target_rate_by_class(
    datablock: pd.DataFrame,
    var_name: str,
    quantiles: Optional[bool] = False
) -> pd.DataFrame:
    """
    Calculate the credit default rate by variable classes.

    This function calculates the credit default rate for each class of
    a specified variable in the given DataFrame.

    Parameters:
    -----------
    datablock : pd.DataFrame
        The pandas DataFrame containing the data.
    var_name : str
        The name of the variable for which to calculate the default rate.
    quantiles : bool, optional
        If True, the variable is binned into quantiles;
        otherwise, it's binned into equally spaced intervals.
        Default is False.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with columns "var_name" and "TARGET_RATE"
        representing the variable classes and their
        corresponding credit default rates.

    Examples:
    ---------
    >>> datablock = pd.DataFrame({'Age': [25, 30, 35, 40, 45], 'TARGET': [0, 1, 0, 0, 1]})
    >>> result = get_target_rate_by_class(datablock, 'Age')
    >>> print(result)
    # Output:
    #    Age  TARGET_RATE
    # 0   25          0.0
    # 1   30          1.0
    # 2   35          0.0
    # 3   40          0.0
    # 4   45          0.5
    """
    # Group by the bin and calculate averages
    x_y = datablock[[var_name, "TARGET"]].copy()
    x_y = x_y[x_y.TARGET > -1]
    x_y.dropna(inplace=True)

    # Bin the data if numeric
    if is_numeric_dtype(x_y[var_name]):
        if quantiles:
            x_y[var_name] = pd.qcut(x_y[var_name], q=10, duplicates="drop")
        else:
            x_y[var_name] = pd.cut(x_y[var_name], bins=10)

    # Group by the class and calculate averages
    grouped_x = x_y.groupby(var_name).mean()

    # Sort by decreasing order if and only if not numeric
    if is_string_dtype(x_y[var_name]):
        grouped_x.sort_values(by="TARGET", ascending=False, inplace=True)

    return grouped_x.reset_index()


def show_target_rate_by_class(
    rate: pd.DataFrame,
    var_name: str,
    ax: Optional[plt.Axes] = None,
    rotate_labels: Optional[bool] = True,
    title: Optional[str] = None
) -> None:
    """
    Show the credit default rate by variable classes in a bar plot.

    This function creates a bar plot to visualize the credit default rate by variable classes.

    Parameters:
    -----------
    rate : pd.DataFrame
        A DataFrame containing the credit default rate for each class of a variable.
    var_name : str
        The name of the variable.
    ax : matplotlib.axes._axes.Axes, optional
        The axes to plot on. If None, a new figure and axes are created.
        Default is None.
    rotate_labels : bool, optional
        Whether to rotate the x-axis tick labels. Default is True.
    title : str, optional
        The title for the plot. If None, a default title is used.
        Default is None.

    Returns:
    --------
    None
    """
    fig = None
    if ax is None:
        figsize = (rate[var_name].nunique(), 4)
        fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(rate, x=var_name, y="TARGET", ax=ax)

    if title is None:
        title = f"Credit Default by `{var_name}` classes"
    ax.set_title(title, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Credit Default (%)")

    # Rotate the x-axis tick labels and set their alignment.
    if rotate_labels:
        plt.setp(
            ax.get_xticklabels(),
            rotation=30, ha="right", rotation_mode="anchor"
    )
    
    # Add percents values above bars
    for p, pct in zip(ax.patches, rate.TARGET):
        ax.annotate(
            f"{100*pct:.1f} %",
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha="center", va="center", fontsize=7, weight="bold", # color="black", 
            xytext=(0, 7), textcoords="offset points"
        )
    
    min_y, max_y = ax.get_ylim()
    ax.set_ylim(min_y, 1.08*max_y)
    
    if fig is not None:
        fig.tight_layout()
        save_and_show(
            f"credit_default_by_class_{title.lower()}",
            sub_dir="dataxplor"
        )


def show_datablock_target_rate_by_class(
    datablock: pd.DataFrame,
    var_renames: Optional[dict] = None,
    quantiles: Optional[bool] = False,
    title: Optional[str] = None
) -> None:
    """
    Show the credit default rate by variable classes in multiple subplots.

    This function creates multiple subplots to visualize the credit default rate by variable classes
    for each variable in the given data block.

    Parameters:
    -----------
    datablock : pd.DataFrame
        A DataFrame containing the data block with a 'TARGET' column.
    var_renames : dict, optional
        A dictionary to rename variables for plotting. Default is None.
    quantiles : bool, optional
        Whether to use quantiles when creating bins for numeric variables. Default is False.
    title : str, optional
        The title for the plot. If None, a default title is used.
        Default is None.

    Returns:
    --------
    None
    """
    var_names = list(datablock.columns)
    var_names.remove("TARGET")
    if var_renames is None:
        var_renames = {}

    # Create the figure and subplots
    ncols = 1
    nrows = len(var_names)
    h = 4
    w = np.max([
        10 if is_numeric_dtype(datablock[var_name])
        else datablock[var_name].nunique()
        for var_name in var_names
    ])

    _, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(w * ncols, h * nrows)
    )
    axes = np.ravel(axes)
    # var_renames = var_renames if var_renames is not None else var_names
    for ax, var_name in zip(axes, var_names):
        rate = get_target_rate_by_class(datablock, var_name, quantiles)
        show_target_rate_by_class(rate, var_name, ax=ax)

    # Add a title to the figure
    if title is None:
        title = "Credit Default by var classes\n"
    plt.suptitle(title, fontsize=15, weight="bold")

    # Adjust the spacing between the subplots and save/show the figure
    plt.tight_layout()
    save_and_show(
        f"{title.lower()}",
        sub_dir="dataxplor"
    )
