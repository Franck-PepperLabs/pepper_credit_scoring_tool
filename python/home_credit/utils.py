from typing import Optional, List, Tuple, Dict

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

from pepper.plots import lin_log_tetra_histplot

from home_credit.load import get_columns_description, get_table


def help_cols(
    col_names: Optional[List[str]] = None,
    table_pat: Optional[str] = None,
    desc_pat: Optional[str] = None,
    spe_pat: Optional[str] = None
) -> None:
    """
    Display column descriptions based on various filters.

    Parameters
    ----------
    col_names : list of str, optional
        A list of column names to filter the descriptions (default is None).
    table_pat : str, optional
        A regular expression pattern to filter the table names (default is None).
    desc_pat : str, optional
        A regular expression pattern to filter the column descriptions (default is None).
    spe_pat : str, optional
        A regular expression pattern to filter the special descriptions (default is None).

    Returns
    -------
    None
    """
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


def get_variables_description() -> pd.DataFrame:
    """
    Get the description of variables used in the Home Credit datasets.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing descriptions of variables.
    """
    descs = get_table("columns_description").copy()
    # Remove _{train|test} suffixes
    descs.Table = descs.Table.str.replace("_{train|test}", "", regex=False)
    return descs


def get_variables_infos() -> pd.DataFrame:
    """
    Get information about variables in the Home Credit datasets.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing information about variables.
    """
    # Create a dictionary of tables, where the table name is the key
    # and the corresponding DataFrame is the value
    table_dict = {
        table_name: get_table(table_name)
        for table_name in get_table_names()
    }
    
    # Calculate discrete statistics for each variable in the tables
    var_dstats = db_discrete_stats(table_dict)
    
    # Get descriptions for each variable
    var_descs = get_variables_description()
    
    # Merge discrete statistics and variable descriptions
    var_infos = pd.merge(
        var_dstats, var_descs,
        left_on=["table_name", "col"],
        right_on=["Table", "Column"]
    )
    
    # Drop unnecessary columns
    var_infos.drop(columns=["Table", "Column"], inplace=True)
    
    # Rename columns for consistency
    var_infos.rename(columns={"table_name": "Table", "col": "Variable"}, inplace=True)

    # Split variable names into individual words
    split_var_names = var_infos.Variable.str.split(pat="_", expand=True)
    split_var_names.columns = [f"WORD_{i}" for i in range(5)]

    # Concatenate variable information and word splits
    return pd.concat([var_infos, split_var_names], axis=1)


def _get_pat(group_name):
    """
    Get a regular expression pattern based on the group name.

    Parameters
    ----------
    group_name : str
        The name of the group for which to get the pattern.

    Returns
    -------
    str
        A regular expression pattern.
    """
    pat_map = {
        "HOUSING_PROFILE": ".*_(AVG|MEDI|MODE)"
    }
    return pat_map.get(group_name)


def get_variables_description_v2(group_name: str) -> pd.DataFrame:
    """
    Get variable descriptions based on a group name.

    Parameters
    ----------
    group_name : str
        The name of the group for which to get variable descriptions.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing variable descriptions.
    """
    # Implement the function to get variable descriptions for the specified group
    # This can be a v2 implementation of the existing get_variables_description
    # based on the group_name provided.
    # You can customize the logic to fetch descriptions for the specific group.
    pass


def help_variables(
    col_pat: Optional[str] = None,
    table_pat: Optional[str] = None,
    desc_pat: Optional[str] = None,
    spe_pat: Optional[str] = None
) -> None:
    """
    Display variable descriptions based on specified patterns.

    Parameters
    ----------
    col_pat : str, optional
        A regular expression pattern to match column names, by default None.
    table_pat : str, optional
        A regular expression pattern to match table names, by default None.
    desc_pat : str, optional
        A regular expression pattern to match descriptions, by default None.
    spe_pat : str, optional
        A regular expression pattern to match special descriptions, by default None.

    Returns
    -------
    None
    """
    # Get the column descriptions data table
    descs = get_variables_description()
    
    # Create a boolean mask that is True for every row
    mask = pd.Series(True, index=descs.index)
    
    # Modify the mask depending on the specified patterns
    if col_pat is not None:
        mask &= descs.Column.str.match(col_pat)
    if table_pat is not None:
        mask &= descs.Table.str.match(table_pat)
    if desc_pat is not None:
        mask &= descs.Description.str.match(desc_pat)
    if spe_pat is not None:
        mask &= descs.Special.str.match(spe_pat)
        
    # Display the filtered variable descriptions in markdown format
    display_dataframe_in_markdown(descs[mask])


def get_table_with_reminder(table_name: str) -> pd.DataFrame:
    """
    Get a table by name and display discrete stats and column descriptions.

    Parameters
    ----------
    table_name : str
        The name of the table to retrieve.

    Returns
    -------
    pd.DataFrame
        The requested table.
    """
    # Retrieve the table data
    table = get_table(table_name)

    # Display discrete statistics for the table
    print_subtitle("Discrete stats")
    display(discrete_stats(table))

    # Display column descriptions for the table
    print_subtitle("Column descriptions")
    help_cols(table_pat=table_name)

    return table



def get_table_names(raw: bool = False) -> List[str]:
    """
    Get a list of table names used in the Home Credit dataset.

    Parameters
    ----------
    raw : bool, optional
        If True, includes raw data tables (default is False).

    Returns
    -------
    List[str]
        A list of table names.
    """
    table_names = [
        "previous_application", "bureau", "bureau_balance",
        "pos_cash_balance", "credit_card_balance", "installments_payments"
    ]
    if raw:
        table_names.extend([
            "application_train", "application_test",
            "columns_description", "sample_submission"
        ])
    else:
        table_names.append("application")

    return table_names


def get_tables_dict(raw: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Get a dictionary of tables from the Home Credit dataset.

    Parameters
    ----------
    raw : bool, optional
        If True, includes raw data tables (default is False).

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary where keys are table names and values are DataFrame objects.
    """
    return {
        table_name: get_table(table_name)
        for table_name in get_table_names(raw)
    }


""" Home Credit business var types
"""


def get_column_types_dist(df: pd.DataFrame) -> List[Tuple[str, int]]:
    """
    Get the count of columns per type (based on the prefix of their name)
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

    return sorted(
        col_types,
        key=lambda x: priority.get(x[0], float("inf")),
        reverse=False,
    )


def display_frame_basic_infos(df: pd.DataFrame) -> None:
    """
    Display basic information about the given DataFrame.

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
    """
    Generate a report of the number of times each value of a foreign key
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
    Generate a report on the relationship between two tables.

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


""" Classification
"""

def get_class_label_name_map() -> Dict[int, str]:
    """
    Get a mapping between class labels and class names.

    Returns:
    --------
    Dict[int, str]
        A dictionary mapping class labels to class names.
    """
    return {-1: "Unknown", 0: "Negative", 1: "Positive"}
