from typing import *

import pandas as pd
import numpy as np

from IPython.display import display

from pepper.utils import (
    print_subtitle,
    discrete_stats,
    display_dataframe_in_markdown
)
from pepper.univar import agg_value_counts
from pepper.feat_eng import reduce_long_tail

from home_credit.load import get_columns_description, get_table
from home_credit.plots import show_cat_mod_counts


def help_cols(col_names=None, table_pat=None, desc_pat=None, spe_pat=None) -> None:
    # Get the column descriptions data table
    descs = get_columns_description()
    # Create a boolean mask that is True for every row
    mask = pd.Series(True, index=descs.index)
    if not(
        col_names is None
        or isinstance(col_names, (list, tuple, np.ndarray, pd.Index, pd.Series))
    ):
        col_names = [col_names]
    if col_names is not None:
        mask &= descs.Column.isin(col_names)
    if table_pat is not None:
        mask &= descs.Table.str.match(table_pat)
    if desc_pat is not None:
        mask &= descs.Description.str.match(desc_pat)
    if spe_pat is not None:
        mask &= descs.Special.str.match(spe_pat)
    display_dataframe_in_markdown(descs[mask])


def get_table_with_reminder(table_name):
    table = get_table(table_name)
    print_subtitle("Discrete stats")
    display(discrete_stats(table))
    print_subtitle("Column descriptions")
    help_cols(table_pat=table_name)
    return table


""" Data blocks NB macros
"""


def get_datablock(data, pat):
    return data[data.columns[data.columns.str.match(pat)]]


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
        "mod": s.mode()[0],
        "var": s.var(),
        "std": s.std(),
        "skew": s.skew(),
        "kurt": s.kurt(),
    })

def expl_analysis_df(df):
    return df.apply(expl_analysis_series)
