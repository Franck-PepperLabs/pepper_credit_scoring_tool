"""This module contains functions to load (and analyze) data tables for the
`Home Credit` Kaggle Challenge.
"""

from typing import Optional, Iterable, Callable, Union, List, Dict
import os
import pandas as pd
# from IPython.display import display

from pepper.cache import Cache
from pepper.env import get_dataset_pqt_dir, get_dataset_csv_dir, get_tmp_dir
from pepper.persist import get_filenames_glob
from pepper.utils import create_if_not_exist, display_file_link
# from pepper.db_utils import set_pk_index, cast_columns


def get_raw_table_names() -> List[str]:
    """
    Get a list of raw table names available in the dataset directory.

    Returns:
    --------
    List[str]
        A list of raw table names (excluding file extensions).
    """
    filenames = get_filenames_glob(get_dataset_csv_dir(), "csv")
    return [filename[:-4] for filename in filenames]


def load_raw_table(
    table_name: str,
    dataset_dir: Optional[str] = None,
    ext: Optional[str] = None
) -> pd.DataFrame:
    """
    Load a raw data table from a file.

    Parameters
    ----------
    table_name : str
        The table name which is the file name (without extension).
    dataset_dir : str, optional
        The directory where the file is located. Defaults to None.
    ext : str, optional
        The extension of the file. Defaults to None.

    Raises
    ------
    ValueError
        If the specified format is not None, "csv" or "pqt".
    FileNotFoundError
        If the specified file does not exist.
    TypeError
        If the loaded data is not a DataFrame.

    Returns
    -------
    pd.DataFrame
        The loaded data table.
    
    Notes
    -----
    If `dataset_dir` or `ext` is None, the function will attempt to use the default
    values specified in the `get_dataset_pqt_dir` and `get_dataset_csv_dir`
    functions respectively.
    """
    if ext not in [None, "csv", "pqt"]:
        raise ValueError(
            f"Invalid extension '{ext}': use `'pqt'`, `'csv'` or `None`."
        )
    ext = ext or "pqt"
    if dataset_dir is None:
        dataset_dir = (
            get_dataset_pqt_dir()
            if ext == "pqt"
            else get_dataset_csv_dir()
        )
    filepath = os.path.join(dataset_dir, f"{table_name}.{ext}")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"The file '{filepath}' does not exist.")
    data = None
    print("load", filepath)
    if ext == "csv":
        data = pd.read_csv(filepath)
    elif ext == "pqt":
        data = pd.read_parquet(filepath)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Loaded data is not a pandas DataFrame.")
    data.columns.name = f"RAW_{table_name.upper()}"
    return data


""" Raw tables with object dtypes
"""


def filter_by_indices(
    data: pd.DataFrame,
    filter_columns: Dict[str, Optional[Iterable]]
) -> pd.DataFrame:
    """
    Filter a data table by multiple indices.

    Parameters
    ----------
    data : pd.DataFrame
        The data table to filter.
    filter_columns : dict
        A dictionary mapping column names to indices.
        Rows in the data table that do not have values in the indices for the
        respective columns will be filtered out.

    Returns
    -------
    pd.DataFrame
        The data table with rows filtered by the indices.
    """
    # Create a boolean mask that is True for every row
    mask = pd.Series(True, index=data.index)

    # Iterate through the columns to filter by, and update the mask
    # if an index is provided
    for column, index in filter_columns.items():
        if index is not None:
            mask &= data[column].isin(index)

    # Return the data table with the rows that match the mask
    return data[mask]


def _load_columns_description() -> pd.DataFrame:
    """
    Load the description of columns from the 'HomeCredit_columns_description'
    table.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing column descriptions with
        columns 'Table', 'Row', and 'Description'.
    """
    table_name = "HomeCredit_columns_description"
    table = load_raw_table(table_name)
    table.columns.name = "col_descs"
    table.set_index(table.columns[0], inplace=True)
    table.index.name = 'col_id'
    # Rename `Row` column in `Column`
    table.rename(columns={"Row": "Column"}, inplace=True)
    # Remove .csv extension
    # POS_CASH => pos_cash
    table.Table = table.Table.str[:-4].str.lower()
    # 'SK_ID_PREV ' => 'SK_ID_PREV ' : strip
    table.Column = table.Column.str.strip()
    # SK_BUREAU_ID => SK_ID_BUREAU
    table.Column[table.Column == "SK_BUREAU_ID"] = "SK_ID_BUREAU"
    # TODO : etc => faire un gros diff pour les identifier
    return table


def _load_sample_submission() -> pd.DataFrame:
    """
    Load the 'sample_submission' table.

    This function is an alias for `load_raw_table("sample_submission")`
    and is provided for convenience. It returns a DataFrame containing
    the data from the 'sample_submission' table.

    Returns
    -------
    pd.DataFrame
        The loaded 'sample_submission' table as a DataFrame.
    """
    return load_raw_table("sample_submission")


def _load_application_test() -> pd.DataFrame:
    """
    Load the 'application_test' table.

    This function is an alias for `load_raw_table("application_test")`
    and is provided for convenience. It returns a DataFrame containing
    the data from the 'application_test' table.

    Returns
    -------
    pd.DataFrame
        The loaded 'application_test' table as a DataFrame.
    """
    return load_raw_table("application_test")


def _load_application_train() -> pd.DataFrame:
    """
    Load the 'application_train' table.

    This function is an alias for `load_raw_table("application_train")`
    and is provided for convenience. It returns a DataFrame containing
    the data from the 'application_train' table.

    Returns
    -------
    pd.DataFrame
        The loaded 'application_train' table as a DataFrame.
    """
    return load_raw_table("application_train")


def _load_previous_application() -> pd.DataFrame:
    """
    Load the 'previous_application' table.

    This function is an alias for `load_raw_table("previous_application")`
    and is provided for convenience. It returns a DataFrame containing
    the data from the 'previous_application' table.

    Returns
    -------
    pd.DataFrame
        The loaded 'previous_application' table as a DataFrame.
    """
    return load_raw_table("previous_application")


def _load_bureau() -> pd.DataFrame:
    """
    Load the 'bureau' table.

    This function is an alias for `load_raw_table("bureau")`
    and is provided for convenience. It returns a DataFrame containing
    the data from the 'bureau' table.

    Returns
    -------
    pd.DataFrame
        The loaded 'bureau' table as a DataFrame.
    """
    return load_raw_table("bureau")


def _load_bureau_balance() -> pd.DataFrame:
    """
    Load the 'bureau_balance' table.

    This function is an alias for `load_raw_table("bureau_balance")`
    and is provided for convenience. It returns a DataFrame containing
    the data from the 'bureau_balance' table.

    Returns
    -------
    pd.DataFrame
        The loaded 'bureau_balance' table as a DataFrame.
    """
    return load_raw_table("bureau_balance")


def _load_credit_card_balance() -> pd.DataFrame:
    """
    Load the 'credit_card_balance' table.

    This function is an alias for `load_raw_table("credit_card_balance")`
    and is provided for convenience. It returns a DataFrame containing
    the data from the 'credit_card_balance' table.

    Returns
    -------
    pd.DataFrame
        The loaded 'credit_card_balance' table as a DataFrame.
    """
    return load_raw_table("credit_card_balance")


def _load_pos_cash_balance() -> pd.DataFrame:
    """
    Load the 'POS_CASH_balance' table.

    This function is an alias for `load_raw_table("POS_CASH_balance")`
    and is provided for convenience. It returns a DataFrame containing
    the data from the 'POS_CASH_balance' table.

    Returns
    -------
    pd.DataFrame
        The loaded 'POS_CASH_balance' table as a DataFrame.
    """
    return load_raw_table("POS_CASH_balance")


def _load_installments_payments() -> pd.DataFrame:
    """
    Load the 'installments_payments' table.

    This function is an alias for `load_raw_table("installments_payments")`
    and is provided for convenience. It returns a DataFrame containing
    the data from the 'installments_payments' table.

    Returns
    -------
    pd.DataFrame
        The loaded 'installments_payments' table as a DataFrame.
    """
    return load_raw_table("installments_payments")


def _load_application() -> pd.DataFrame:
    """
    Load and combine the 'application_train' and 'application_test' tables.

    This function loads the 'application_train' and 'application_test' tables,
    combines them, and adds a 'TARGET' column with a value of -1 for the test set.
    The resulting DataFrame contains data from both tables.

    Returns
    -------
    pd.DataFrame
        The combined 'application' table as a DataFrame.
    """
    application_train = load_raw_table("application_train")
    application_test = load_raw_table("application_test")
    application_test.insert(1, "TARGET", -1)
    application = pd.concat(
        [application_train, application_test],
        axis=0, ignore_index=True
    )
    application.sort_index(inplace=True)
    application.columns.name = "application"
    return application


def _load_var_descs() -> pd.DataFrame:
    """
    Load variable descriptions from the 'HomeCredit_columns_description' table.

    This function loads the variable descriptions from the
    'HomeCredit_columns_description' table and processes them to create a
    DataFrame with variable descriptions.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing variable descriptions with columns:
        'Table', 'Row', 'Description'.

    Notes
    -----
    The 'Table' column represents the table to which the variable belongs.
    The 'Row' column represents the variable name.
    The 'Description' column contains the description of the variable.
    """
    col_descs = _load_columns_description()
    var_descs = col_descs.copy()
    # Remove metadata : useless, already done by Home Credit
    """var_descs = var_descs[
        ~cols_desc.Table.isin([
            "sample_submission",
            "HomeCredit_columns_description"
        ])
    ]"""
    # Rename `application_train` in `application`
    var_descs.Table[var_descs.Table == "application_{train|test}"] = "application"
    return var_descs


def get_columns_description() -> pd.DataFrame:
    return Cache.init("columns_description", _load_columns_description)


def get_sample_submission() -> pd.DataFrame:
    return Cache.init("sample_submission", _load_sample_submission)


def get_application_test() -> pd.DataFrame:
    return Cache.init("application_test", _load_application_test)


def get_application_train() -> pd.DataFrame:
    return Cache.init("application_train", _load_application_train)


def get_application() -> pd.DataFrame:
    return Cache.init("application", _load_application)


def get_bureau() -> pd.DataFrame:
    return Cache.init("bureau", _load_bureau)


def get_bureau_balance() -> pd.DataFrame:
    return Cache.init("bureau_balance", _load_bureau_balance)


def get_credit_card_balance() -> pd.DataFrame:
    return Cache.init("credit_card_balance", _load_credit_card_balance)


def get_pos_cash_balance() -> pd.DataFrame:
    return Cache.init("pos_cash_balance", _load_pos_cash_balance)


def get_installments_payments() -> pd.DataFrame:
    return Cache.init("installments_payments", _load_installments_payments)


def get_previous_application() -> pd.DataFrame:
    return Cache.init("previous_application", _load_previous_application)


def get_var_descs() -> pd.DataFrame:
    return Cache.init("var_descs", _load_var_descs)


def get_target() -> pd.DataFrame:
    """DEPRECATED User home_credit.tables.to_target_map instead"""
    tgt = get_application()[["SK_ID_CURR", "TARGET"]].set_index("SK_ID_CURR")
    return Cache.init("target", lambda: tgt)


def get_main_map() -> pd.DataFrame:
    """
    Generate a table of client correspondences.

    This function constructs a table containing the correspondences for 'SK_ID_CURR', 'SK_ID_BUREAU',
    'SK_ID_PREV', and 'TARGET' from multiple source tables. The resulting table is used to provide
    data for client selection in the application for mission officers.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ['SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'TARGET'].

    Notes
    -----
    The function loads the 'application', 'bureau', and 'previous_application' tables, selects
    the necessary columns, and merges them based on 'SK_ID_CURR'.
    """
    def _get_main_map() -> pd.DataFrame:
        # Load the three tables
        application = get_table("application")
        bureau = get_table("bureau")
        previous_application = get_table("previous_application")

        # Select only the necessary columns in each table
        application = application[["SK_ID_CURR", "TARGET"]]
        bureau = bureau[["SK_ID_CURR", "SK_ID_BUREAU"]]
        previous_application = previous_application[["SK_ID_CURR", "SK_ID_PREV"]]

        # Merge the 'application' and 'bureau' tables on 'SK_ID_CURR'
        data = pd.merge(application, bureau, on="SK_ID_CURR", how="inner")

        # Merge the previously merged table with 'previous_application' on 'SK_ID_CURR'
        data = pd.merge(data, previous_application, on="SK_ID_CURR", how="inner")
        
        # TODO : on le fait ici sous la pression des circonstances
        # mais c'est à remonter plus en amont avec traitement systématique
        # sur la base des informations issues de l'analyse exploratoire
        data.TARGET = data.TARGET.astype("int8")
        data.SK_ID_CURR = data.SK_ID_CURR.astype("int32")
        data.SK_ID_BUREAU = data.SK_ID_BUREAU.astype("int32")
        data.SK_ID_PREV = data.SK_ID_PREV.astype("int32")
        
        return data[["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "TARGET"]]

    return Cache.init("main_map", _get_main_map)



def get_table_loaders_dict() -> Dict[str, Callable]:
    """
    Get a dictionary of table loaders for various tables.

    Returns
    -------
    Dict[str, Callable]
        A dictionary mapping table names to functions that load the corresponding table.

    Notes
    -----
    This function returns a dictionary where the keys are the names of tables,
    and the values are callable functions that can be used to load the respective tables.
    """
    return {
        # Raw tables
        "columns_description": get_columns_description,
        "sample_submission": get_sample_submission,
        "application_test": get_application_test,
        "application_train": get_application_train,
        "previous_application": get_previous_application,
        "bureau": get_bureau,
        "bureau_balance": get_bureau_balance,
        "credit_card_balance": get_credit_card_balance,
        "pos_cash_balance": get_pos_cash_balance,
        "installments_payments": get_installments_payments,
        # Derived tables
        "application": get_application,
        "var_descs": get_var_descs,
        "target": get_target,
        "main_map": get_main_map,
    }



def get_table(table_name: str) -> pd.DataFrame:
    """
    Get a specified table by name.

    Parameters
    ----------
    table_name : str
        The name of the table to retrieve.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the specified table's data.

    Raises
    ------
    ValueError
        If the specified `table_name` is not found in the available tables.
    """
    table_loaders_dict = get_table_loaders_dict()
    if table_name in table_loaders_dict:
        return table_loaders_dict[table_name]()
    else:
        raise ValueError(
            f"Unknown `table_name` {table_name}"
        )


def get_prep_dataset_dir() -> str:
    """
    Return the preprocessed dataset versions path.

    Returns
    -------
    str
        The project's preprocessed dataset versions path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.
    """
    return os.path.join(get_tmp_dir(), "prep_dataset")


# Saves the preprocessed dataset version
def save_prep_dataset(data: pd.DataFrame, version_name: str) -> pd.DataFrame:
    """Save a preprocessed dataset version to disk."""
    prep_dataset_dir = get_prep_dataset_dir()
    create_if_not_exist(prep_dataset_dir)
    filepath = os.path.join(
        prep_dataset_dir,
        f"prep_dataset_{version_name}.pqt"
    )
    data.to_parquet(filepath, engine="pyarrow", compression="gzip")


# Loads the preprocessed dataset version
def load_prep_dataset(version_name: str) -> pd.DataFrame:
    """Load a preprocessed dataset version from disk."""
    prep_dataset_dir = get_prep_dataset_dir()
    filepath = os.path.join(
        prep_dataset_dir,
        f"prep_dataset_{version_name}.pqt"
    )
    return pd.read_parquet(filepath, engine="pyarrow")


def _get_tmp_subdir(dir_name: str) -> str:
    """Return the path for a temporary subdirectory."""
    tmp_subdir = os.path.join(get_tmp_dir(), dir_name)
    create_if_not_exist(tmp_subdir)
    return tmp_subdir 

def get_submission_dir() -> str:
    """Return the path for submission files."""
    return _get_tmp_subdir("submission")


def get_mlflow_dir() -> str:
    """Return the path for MLflow logs."""
    return _get_tmp_subdir("mlruns")

def get_reports_dir() -> str:
    """Return the path for report files."""
    return _get_tmp_subdir("reports")


def save_submission(sms_data: pd.DataFrame, sms_filename: str):
    """Save a submission file to disk."""
    submission_dir = get_submission_dir()
    create_if_not_exist(submission_dir)
    filepath = os.path.join(submission_dir, sms_filename)
    sms_data.to_csv(filepath, index=False)
    display_file_link(filepath, "<b>Submission file</b> saved 🔗 ")
