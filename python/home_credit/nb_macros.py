#from typing import List, Tuple, Dict

from IPython.display import display

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

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


def get_datablock(data, pat):
    if isinstance(data, str):
        data = get_table(data).copy()
    return data[data.columns[data.columns.str.match(pat)]]


def get_labeled_datablock(table_name, pat):
    data = get_table(table_name).copy()
    if "TARGET" not in data.columns:
        target = targetize(data.SK_ID_CURR)
        data.insert(0, "TARGET", target)
    return get_datablock(data, f"{pat}|TARGET")


def var_catdist_report(s, agg=True):
    print_subtitle(s.name)
    help_cols(s.name)
    avc = agg_value_counts(s, agg=agg)
    display(avc)
    r = reduce_long_tail(s, agg=agg)
    show_cat_mod_counts(r, order=avc.index) 


def datablock_catdist_report(data, agg=True):
    for cat in data.columns:
        var_catdist_report(data[cat], agg=agg)


import pandas as pd
pd.options.display.float_format = "{:.3f}".format
def expl_analysis_series(s):
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

def expl_analysis_df(df):
    return df.apply(expl_analysis_series)


""" Credit default rate by a var classes
"""


def get_target_rate_by_class(
    datablock: pd.DataFrame,
    var_name: str,
    quantiles: bool = False
) -> pd.DataFrame:
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
    # order=None,
    ax=None,
    rotate_labels=True,
    title=None
):
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
        save_and_show(f"credit_default_by_class_{title.lower()}", sub_dir="dataxplor")


def show_datablock_target_rate_by_class(
    datablock,
    var_renames=None,
    quantiles: bool = False,
    # transf_var: Optional[callable] = None,
    # single_figsize: Tuple[int, int] = (8, 3),
    title=None
):
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
    save_and_show(f"{title.lower()}", sub_dir="dataxplor")
