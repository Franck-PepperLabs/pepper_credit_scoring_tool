from typing import List, Tuple, Dict

import pandas as pd
import numpy as np

from IPython.display import display

from pepper.utils import (
    bold,
    print_subtitle,
    discrete_stats,
    display_dataframe_in_markdown
)
from pepper.pd_utils import subindex
from pepper.db_utils import (
    db_discrete_stats,
    out_of_intersection,
    display_is_in_A_but_not_in_B_heatmaps
)

# from pepper.univar import agg_value_counts
# from pepper.feat_eng import reduce_long_tail
from pepper.plots import lin_log_tetra_histplot   # show_cat_mod_counts, 

from home_credit.load import get_columns_description, get_table
# from home_credit.merge import targetize


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


def get_variables_description():
    descs = get_table("columns_description").copy()
    # Remove _{train|test} suffixes
    descs.Table = descs.Table.str.replace("_{train|test}", "", regex=False)
    return descs


def get_variables_infos():
    table_dict = {
        table_name: get_table(table_name)
        for table_name in get_table_names()
    }
    var_dstats = db_discrete_stats(table_dict)
    var_descs = get_variables_description()
    var_infos = pd.merge(
        var_dstats, var_descs,
        left_on=["table_name", "col"],
        right_on=["Table", "Column"]
    )
    var_infos.drop(columns=["Table", "Column"], inplace=True)
    var_infos.rename(columns={"table_name": "Table", "col": "Variable"}, inplace=True)

    split_var_names = var_infos.Variable.str.split(pat="_", expand=True)
    split_var_names.columns = [f"WORD_{i}" for i in range(5)]

    return pd.concat([var_infos, split_var_names], axis=1)


def _get_pat(group_name):
    pat_map = {
        "HOUSING_PROFILE": ".*_(AVG|MEDI|MODE)"
    }
    return pat_map.get(group_name)

def get_variables_description(group_name):
    pat = _get_pat(group_name)


def help_variables(col_pat=None, table_pat=None, desc_pat=None, spe_pat=None) -> None:
    # Get the column descriptions data table
    descs = get_variables_description()
    # Create a boolean mask that is True for every row
    mask = pd.Series(True, index=descs.index)
    # Modify the mask depending on pats
    if col_pat is not None:
        mask &= descs.Column.str.match(col_pat)
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


def get_table_names():
    return [
        "application", "bureau", "previous_application",
        "bureau_balance", "pos_cash_balance",
        "credit_card_balance", "installments_payments"
    ]


""" Home Credit business var types
"""


def get_column_types_dist(df: pd.DataFrame) -> List[Tuple[str, int]]:
    """Gets the count of columns per type (based on the prefix of their name)
    in the given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to analyze.

    Returns
    -------
    List[Tuple[str, int]]
        List of tuples representing the count of columns per type.
        Each tuple contains a string representing the type and an integer
        representing the count.
        The list is sorted in priority order.
    """
    # Define column types sorting priority
    priority = {
        "SK": 0, "FLAG":1, "NFLAG":2, "NAME": 3, "NUM":4,
        "DAYS":5, "CNT": 6, "AMT": 7, "MONTHS": 8
    }

    # Get column types counts
    col_types = list(zip(
        *np.unique(
            [
                col[:col.index("_")] if "_" in col else col
                for col in df.columns
            ],
            return_counts=True
        )
    ))

    # Sort it based on priority
    sorted_col_types = sorted(
        col_types,
        key=lambda x: priority.get(x[0], float("inf")),
        reverse=False
    )

    return sorted_col_types


def display_frame_basic_infos(df: pd.DataFrame) -> None:
    """Displays basic information about the given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to display information about.

    Returns
    -------
    None
        This function doesn't return anything, it just prints to the console.

    Prints
    ------
    str
        A formatted string containing the number of samples (rows) in the
        DataFrame.
    str
        A formatted string containing the number of columns in the DataFrame,
        as well as a list of tuples representing the count of columns per type.
        Each tuple contains a string representing the type and an integer
        representing the count. The list is sorted in priority order.
    """
    print(f"{bold('n_samples')}: {df.shape[0]:n}")
    print(f"{bold('n_columns')}: {df.shape[1]:n}, {get_column_types_dist(df)}")


def foreign_key_counts_report(table_name: str, sk_name: str) -> None:
    """Generates a report of the number of times each value of a foreign key
    appears in a table. Displays a summary of the counts and a histogram of
    their distribution.

    Parameters
    ----------
    table_name : str
        The name of the table to analyze.
    sk_name : str
        The name of the foreign key in the table.

    Returns
    -------
    None
    """
    data = get_table(table_name)
    sk_id = data[sk_name]
    subidx = subindex(sk_id).subindex
    sk_id_counts = subidx + 1
    display(pd.DataFrame(sk_id_counts.describe()).T.astype(int))
    lin_log_tetra_histplot(
        sk_id_counts,
        title=f"`{table_name}`:`{sk_name}` counts histogram"
    )


def main_subs_relation_report(
    main_table_name: str,
    subs_table_name: str,
    sk_name: str
) -> None:
    """
    Generates a report on the relationship between two tables.

    Parameters
    ----------
    main_table_name : str
        The name of the main table to be analyzed.
    subs_table_name : str
        The name of the subsidiary table to be analyzed.
    sk_name : str
        The name of the key which is the primary key in the main table and the 
        foreign key in the subsidiary one.

    Returns
    -------
    None
        This function does not return anything.
    """
    _ = display_is_in_A_but_not_in_B_heatmaps({
        main_table_name: get_table(main_table_name),
        subs_table_name: get_table(subs_table_name)
    }, sk_name)


""" Relational
"""


def _not_in_subs_table_idx(
    main_table_name: str,
    subs_table_name: str,
    key: str
) -> np.ndarray:
    """
    Returns an array of primary keys that are not present in a subsidiary
    table.

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
    np.ndarray
        An array of primary keys that are not present in the subsidiary table.
    """

    # Find untracked primary keys from the main table compared to the subset
    # table based on the key 
    _, _, untracked_sk_id, _ = out_of_intersection(
        get_table(main_table_name),
        get_table(subs_table_name),
        key
    )
    return untracked_sk_id


""" Classif
"""

def get_class_label_name_map() -> Dict[int, str]:
    """Gets a mapping between class labels and class names.

    Returns:
    --------
    Dict[int, str]
        A dictionary mapping class labels to class names.
    """
    #return dict(enumerate(get_product_categories().level_0.cat.categories)) [in P6]
    return {-1: "Unknown", 0: "Negative", 1: "Positive"}
