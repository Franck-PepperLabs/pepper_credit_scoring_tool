"""This module contains functions to load (and analyze) data tables for the
`Home Credit` Kaggle Challenge.
"""

from typing import Optional, List, Union, Dict, Iterable
import os
import pandas as pd
# from IPython.display import display

from pepper.cache import Cache
from pepper.env import get_dataset_pqt_dir, get_dataset_csv_dir, get_tmp_dir
from pepper.persist import _get_filenames_glob
from pepper.utils import create_if_not_exist, display_file_link
# from pepper.db_utils import set_pk_index, cast_columns


def get_raw_table_names() -> List[str]:
    filenames = _get_filenames_glob(get_dataset_csv_dir(), "csv")
    return [filename[:-4] for filename in filenames]


def load_raw_table(
    table_name: str,
    dataset_dir: Optional[str] = None,
    ext: Optional[str] = None
) -> pd.DataFrame:
    """Loads a raw data table from a file.

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
    """Filters a data table by multiple indices.

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
    return load_raw_table("sample_submission")


def _load_application_test() -> pd.DataFrame:
    return load_raw_table("application_test")


def _load_application_train() -> pd.DataFrame:
    return load_raw_table("application_train")


def _load_previous_application() -> pd.DataFrame:
    return load_raw_table("previous_application")


def _load_bureau() -> pd.DataFrame:
    return load_raw_table("bureau")


def _load_bureau_balance() -> pd.DataFrame:
    return load_raw_table("bureau_balance")


def _load_credit_card_balance() -> pd.DataFrame:
    return load_raw_table("credit_card_balance")


def _load_pos_cash_balance() -> pd.DataFrame:
    return load_raw_table("POS_CASH_balance")


def _load_installments_payments() -> pd.DataFrame:
    return load_raw_table("installments_payments")


def _load_application() -> pd.DataFrame:
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
    tgt = get_application()[["SK_ID_CURR", "TARGET"]].set_index("SK_ID_CURR")
    return Cache.init("target", lambda: tgt)


def get_table(table_name: str) -> pd.DataFrame:
    table_loaders_dict = {
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
    }
    if table_name in table_loaders_dict:
        return table_loaders_dict[table_name]()
    else:
        raise ValueError(
            f"Unknown `table_name` {table_name}"
        )


def get_prep_dataset_dir() -> str:
    """Returns the preprocessed dataset versions path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.

    Returns
    -------
    str
        The project's preprocessed dataset versions path.
    """
    return os.path.join(get_tmp_dir(), "prep_dataset")


# Saves the preprocessed dataset version
def save_prep_dataset(data: pd.DataFrame, version_name: str) -> pd.DataFrame:
    prep_dataset_dir = get_prep_dataset_dir()
    create_if_not_exist(prep_dataset_dir)
    filepath = os.path.join(
        prep_dataset_dir,
        f"prep_dataset_{version_name}.pqt"
    )
    data.to_parquet(filepath, engine="pyarrow", compression="gzip")


# Loads the preprocessed dataset version
def load_prep_dataset(version_name: str) -> pd.DataFrame:
    prep_dataset_dir = get_prep_dataset_dir()
    filepath = os.path.join(
        prep_dataset_dir,
        f"prep_dataset_{version_name}.pqt"
    )
    return pd.read_parquet(filepath, engine="pyarrow")


def _get_tmp_subdir(dir_name: str) -> str:
    dir = os.path.join(get_tmp_dir(), dir_name)
    create_if_not_exist(dir)
    return dir 

def get_submission_dir() -> str:
    return _get_tmp_subdir("submission")


def get_mlflow_dir() -> str:
    return _get_tmp_subdir("mlruns")

def get_reports_dir() -> str:
    return _get_tmp_subdir("reports")


def save_submission(sms_data: pd.DataFrame, sms_filename: str):
    submission_dir = get_submission_dir()
    create_if_not_exist(submission_dir)
    filepath = os.path.join(submission_dir, sms_filename)
    sms_data.to_csv(filepath, index=False)
    display_file_link(filepath, "<b>Submission file</b> saved 🔗 ")
