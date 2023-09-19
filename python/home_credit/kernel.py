# Inspired bu and derived from :
# https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features
# kaggle kernels output jsaguiar/lightgbm-with-simple-features -p /path/to/dest

# Common Python foundations imports
from typing import Optional, Union, List, Tuple
import re

import time
from contextlib import contextmanager

# Common datalibs imports
import numpy as np
import pandas as pd
from IPython.display import display

# Don't remove it! : mandatory for `add_derived_features``
from numpy import where

# Imports for `display_importance``
import matplotlib.pyplot as plt
import seaborn as sns

# Imports for `kfold_lightgbm`
import lightgbm as lgbm
# from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

import warnings

"""**Note** Never use this practice, it's a very bad habit!

>>> import warnings
>>> warnings.filterwarnings(action="ignore", category=UserWarning)
>>> warnings.simplefilter(action="ignore", category=FutureWarning)

Instead, prefer targeted muting, for instance:

>>> warnings.filterwarnings(
>>>     "ignore", message=".'verbose' argument is deprecated.",
>>>     category=DeprecationWarning
>>> )
"""

# Home imports
from pepper.debug import tx, kv, tl, stl, sstl, this_f_name
from pepper.utils import save_and_show, pretty_timedelta_str

from home_credit.utils import get_table
from home_credit.load import save_submission

from home_credit.lightgbm_kernel import one_hot_encoder  # For compatibility check


@contextmanager
def exec_tracking(
    title: str,
    context: str,
    inputs: dict,
    outputs: dict,
    verbosity: int
) -> None:
    """Context manager to track code blocks (time, etc)."""
    tl(verbosity, f"{title} - {context}")
    t_0 = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - t_0
        outputs['elapsed_time'] = elapsed_time


        if verbosity is None or verbosity == 0:
            return

        kv(2, "elapsed time", f"{pretty_timedelta_str(elapsed_time, 2)}")

        if "Preprocess" in title:
            adj_table_name = inputs.get("adj_table_name")
            adj_table = outputs.get("adj_table")
            kv(verbosity, f"`{adj_table_name}` shape:", f"{adj_table.shape}")

            if verbosity > 1:
                data = inputs.get("data")
                updated_data = outputs.get("updated_data")
                kv(verbosity, f"\ndata shape", f"{data.shape}")
                data.info(max_cols=1)
                kv(verbosity, f"\nadj_table shape", f"{adj_table.shape}")
                adj_table.info(max_cols=1)
                kv(verbosity, f"\nupdated_data shape", f"{updated_data.shape}")
                updated_data.info(max_cols=1)

            if verbosity > 2:
                stl(verbosity, "inputs")
                for k, v in inputs.items():
                    if isinstance(v, (int, float, bool, str)):
                        kv(verbosity, k, v)
                    else:
                        sstl(verbosity, k)
                        if isinstance(v, pd.DataFrame):
                            v.info(max_cols=1)
                        else:
                            display(v)
                        
                stl(verbosity, "outputs")
                for k, v in inputs.items():
                    if isinstance(v, (int, float, bool, str)):
                        kv(verbosity, k, v)
                    else:
                        sstl(verbosity, k)
                        if isinstance(v, pd.DataFrame):
                            v.info(max_cols=1)
                        else:
                            display(v)


def load_data(
    table_name: str,
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load a table from cache and sample rows if `nrows` is specified.

    Parameters
    ----------
    table_name : str
        The name of the table to load.
    nrows : int, optional
        The number of rows to sample from the loaded data.
        If None, return all data. Default is None.

    Returns
    -------
    pd.DataFrame
        The loaded and sampled data.

    """
    data = get_table(table_name).copy()
    return data if nrows is None else data.sample(nrows)


def get_categorical_vars(
    df: pd.DataFrame,
    dtype: type = object,
    max_modalities: int = None
) -> list:
    """
    Get a list of categorical variables in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to search for categorical variables.
    dtype : type, optional
        The data type(s) to search for. If None, all columns will be searched.
        If a single type is given, only columns of that type will be returned.
        If a list or tuple of types is given, columns of any of those types
        will be returned. Defaults to object.
    max_modalities : int, optional
        The maximum number of unique values a column can have to be considered
        categorical. If None, all columns that match the dtype criteria will
        be returned. Defaults to None.

    Returns
    -------
    list
        A list of column names that meet the criteria for categorical variables.

    """
    if dtype is None and max_modalities is None:
        # Return all columns if no dtype or max_modalities is specified
        return list(df.columns)
    elif dtype is None:
        # Return columns with unique values less than or equal to max_modalities
        return [col for col in df.columns if df[col].nunique() <= max_modalities]
    elif max_modalities is None:
        if isinstance(dtype, (list, tuple, np.ndarray, pd.Index, pd.Series)):
            # Return columns with data types that match any of the given types
            return [col for col in df.columns if df[col].dtype in dtype]
        else:
            # Return columns with the specified data type
            return [col for col in df.columns if df[col].dtype == dtype]
    elif isinstance(dtype, (list, tuple, np.ndarray, pd.Index, pd.Series)):
        # Return columns with data types that match any of the given types and
        # have unique values less than or equal to max_modalities
        return [
            col for col in df.columns
            if (
                df[col].dtype in dtype
                and df[col].nunique() <= max_modalities
            )
        ]
    else:
        # Return columns with the specified data type and
        # have unique values less than or equal to max_modalities
        return [
            col for col in df.columns
            if (
                df[col].dtype == dtype
                and df[col].nunique() <= max_modalities
            )
        ]


def one_hot_encode_all_cats(
    data: pd.DataFrame,
    columns: list[str] = None,
    dummy_na: bool = True,
    drop_first: bool = True,
    dtype: type = np.int8,
    # sparse: bool = True,  # Future : pb with groupby -> scipy ooc_matrix, etc
    discard_constants: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode all categorical columns in the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame)
        The DataFrame to be encoded.
    columns : list[str], optional
        The list of column names to be encoded.
        If None, all categorical columns will be encoded. Defaults to None.
    dummy_na : bool, optional
        Add a column to indicate NaNs. Defaults to True.
    drop_first : bool, optional
        Whether to get k-1 dummies out of k categorical levels by removing the
        first level. Defaults to True.
    dtype : type, optional
        The data type of the encoded columns. Defaults to `np.int8`.
    sparse : bool, optional
        Whether to return a sparse matrix or not. Defaults to True.
    discard_constants : bool, optional
        Whether to remove the potentially generated constant columns
        by `dummy_na`. Defaults to True.

    Returns
    -------
    pd.DataFrame
        A tuple containing the one-hot encoded DataFrame and a list of the
        names of the categorical variables.

    """
    ohe_data = pd.get_dummies(
        data, columns=columns, dummy_na=dummy_na,
        drop_first=drop_first, dtype=dtype,  # Future : sparse=sparse
    )

    # Remove the constant columns potentially generated by `dummy_na`
    if discard_constants:
        const_cols = ohe_data.apply(pd.Series.nunique)
        const_cols = const_cols[const_cols == 1]
        ohe_data.drop(columns=const_cols.index, inplace=True)
    
    # Calculate the new columns index
    ohe_columns = ohe_data.columns.difference(data.columns, sort=False)
    
    # Cast the new columns in bool
    # Warning : with a left join introducing nans, the bool or int8 dtypes becomes object
    for c in ohe_columns:
        ohe_data[c] = ohe_data[c].astype(bool)

    return ohe_data, list(ohe_columns)


# drop_first=False : Only for iso-functionality validation : silly
def hot_encode_cats(
    data: pd.DataFrame,
    dummy_na: bool = True,
    drop_first: bool = False,
    discard_constants: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode categorical features in the input DataFrame and returns
    the resulting DataFrame along with a list of the names of the categorical
    variables.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    dummy_na : bool, optional
        Whether or not to create a dummy column for missing values.
        Defaults to True.
    drop_first : bool, optional
        Whether or not to drop the first column to avoid multi-colinearity.
        Defaults to False.
    discard_constants : bool, optional
        Whether to remove the potentially generated constant columns
        by `dummy_na`. Defaults to True.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        A tuple containing the one-hot encoded DataFrame and a list of the
        names of the categorical variables.
    """
    # Get the names of the categorical variables
    catvar_names = get_categorical_vars(data)

    # One-hot encode all categorical variables
    ohe_data, dumvar_names = one_hot_encode_all_cats(
        data, catvar_names,
        dummy_na=dummy_na,
        drop_first=drop_first,
        discard_constants=discard_constants
    )

    return ohe_data, dumvar_names


def add_derived_features(
    data: pd.DataFrame,
    expr: str
) -> None:
    # sourcery skip: pandas-avoid-inplace
    """
    Add new derived features to a given DataFrame by evaluating an
    expression.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to which the derived features will be added.
    expr : str
        A string containing the mathematical expression that defines the new features.
        The expression must use the column names in the DataFrame as variables and can include
        basic arithmetic operations (+, -, *, /), comparison operators (==, <, >), and logical
        operators (and, or, not). It can also use the `@where` function to implement conditions.

    Returns
    -------
    None

    Example
    -------
    >>> expr = '''
    >>> PAYMENT_PERC = AMT_PAYMENT / AMT_INSTALMENT   # Percentage paid in each installment
    >>> PAYMENT_DIFF = AMT_INSTALMENT - AMT_PAYMENT   # Difference paid in each installment
    >>> DPD = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT    # Days past due (no negative values)
    >>> DBD = DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT    # Days before due (no negative values)
    >>> DPD = @where(DPD > 0, DPD, 0)
    >>> DBD = @where(DBD > 0, DBD, 0)
    >>> '''
    >>> add_derived_features(data, expr)
    """
    # Use numexpr to evaluate the expression
    if not expr:
        return
    data.eval(expr, inplace=True, engine="numexpr")


def flatten_and_prefix_columns(data: pd.DataFrame, prefix: str = None) -> None:
    """
    Flatten MultiIndex columns and adds a prefix to all columns in the
    given DataFrame.
    
    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame whose columns will be flattened and prefixed.
    prefix : str, optional
        The string prefix to add to each column name. Defaults to None.

    """
    # Generate new column names by concatenating prefix, level 0, and level 1
    if prefix is None or not prefix:
        prefix = ""
    else:
        prefix += "_"
    new_columns = [f"{prefix}{c[0]}_{c[1].upper()}" for c in data.columns]
    # Assign new column names to DataFrame columns
    data.columns = pd.Index(new_columns)


def display_importances(feat_imp: pd.DataFrame, title: str = None) -> pd.DataFrame:
    """
    Display a bar plot of feature importance.

    Parameters
    ----------
    feat_imp : pd.DataFrame)
        A dataframe with feature importance information.
    title : str, optional
        The title of the plot. Defaults to None.

    Returns
    -------
    The selected and ordered best features
    """
    # Group and sort the features by importance
    best_features = (
        feat_imp[["feature", "importance"]]
        .groupby("feature").mean()
        .sort_values(by="importance", ascending=False)
    )
    best_features.reset_index(inplace=True)

    # Shorten feature names longer than 30 characters
    best_features.feature = best_features.feature.apply(
        lambda x: x if len(x) <= 30 else f"{x[:15]}(...){x[-15:]}"
    )

    # Create the plot of the 40 best features
    plt.figure(figsize=(8, 10))
    ax = sns.barplot(x="importance", y="feature", data=best_features[:40])

    # Set the title
    title = "" if title is None else title
    plt.title(
        f"{title}{' ' if title else ''}Features importance (avg over folds)",
        fontsize=14,
        fontweight="bold",
    )

    # Remove the labels from the axes
    plt.xlabel("")
    plt.ylabel("")

    # Duplicate the xticks label to the top of the figure
    ax.xaxis.set_tick_params(labeltop=True, labelbottom=True)
    ax.tick_params(axis="x", labelsize=10)

    # Make the grid lines as light as possible
    plt.grid(axis="x", linestyle=":", alpha=0.5)

    # Adjust the spacing between the subplots and save/show the figure
    # plt.subplots_adjust(left=0.4)
    plt.tight_layout()

    # Save and show the figure
    save_and_show(f"feature_importace_{title.lower()}", sub_dir="feat_imp")

    return best_features


""" V1 vs V2 vs Vf
"""


def preprocessed_application(
    nrows=None,
    dummy_na=True,
    drop_first=True
) -> pd.DataFrame:
    # Voir application_train_test_v1 et application_train_test_v2
    data = load_data("application", nrows)
    clean_A_data(data)
    data, _ = hot_encode_cats(data, dummy_na=dummy_na, drop_first=drop_first)
    # add_A_derived_features(data)
    add_derived_features(data,
        """
        DAYS_EMPLOYED_PERC = DAYS_EMPLOYED / DAYS_BIRTH
        INCOME_CREDIT_PERC = AMT_INCOME_TOTAL / AMT_CREDIT
        INCOME_PER_PERSON = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS
        ANNUITY_INCOME_PERC = AMT_ANNUITY / AMT_INCOME_TOTAL
        PAYMENT_RATE = AMT_ANNUITY / AMT_CREDIT
        """
    )
    return data


def preprocessed_bureau_and_balance(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    data = load_data("bureau", nrows)
    adj_data = load_data("bureau_balance")
    if nrows is not None:
        adj_data = adj_data[adj_data.SK_ID_BUREAU.isin(data.SK_ID_BUREAU)]
    clean_B_data(data)
    data, catvars = hot_encode_cats(data, nan_as_category)
    adj_data, adj_catvars = hot_encode_cats(adj_data, nan_as_category)
    add_B_derived_features(data)
    data = group_BB_by_bur_and_join_to_B_v2(data, adj_data, adj_catvars)
    return group_B_by_curr_v2(data, catvars, adj_catvars)


def preprocessed_previous_application(
    nrows=None,
    dummy_na=True,
    drop_first=True
) -> pd.DataFrame:
    data = load_data("previous_application", nrows)
    clean_PA_data(data)
    data, cats = hot_encode_cats(data, dummy_na=dummy_na, drop_first=drop_first)
    add_PA_derived_features(data)
    return group_PA_by_curr_v2(data, cats)


def preprocessed_pos_cash_balance(
    nrows=None,
    dummy_na=True,
    drop_first=True
) -> pd.DataFrame:
    data = load_data("pos_cash_balance", nrows)
    clean_PCB_data(data)
    data, cats = hot_encode_cats(data, dummy_na=dummy_na, drop_first=drop_first)
    add_PCB_derived_features(data)
    return group_PCB_by_curr_v2(data, cats)


def preprocessed_credit_card_balance(
    nrows=None,
    dummy_na=True,
    drop_first=True
) -> pd.DataFrame:
    data = load_data("credit_card_balance", nrows)
    clean_CCB_data(data)
    data, cats = hot_encode_cats(data, dummy_na=dummy_na, drop_first=drop_first)
    add_CCB_derived_features(data)
    return group_CCB_by_curr_v2(data, cats)


def preprocessed_installments_payments(
    nrows=None,
    dummy_na=True,
    drop_first=True
) -> pd.DataFrame:
    data = load_data("installments_payments", nrows)
    clean_IP_data(data)
    data, cats = hot_encode_cats(data, dummy_na=dummy_na, drop_first=drop_first)
    add_IP_derived_features(data)
    return group_IP_by_curr_v2(data, cats)


def preprocess_table(
    table_name: str,
    nrows=None,
    dummy_na=True,
    drop_first=True,
    clean_expr=None,
    derived_features_expr=None,
    agg_rules=None
) -> pd.DataFrame:
    data = load_data(table_name, nrows)
    clean_data(data, clean_expr)
    data, cats = hot_encode_cats(data, dummy_na=dummy_na, drop_first=drop_first)
    add_derived_features(data, derived_features_expr)
    return group_by_curr(data, cats, agg_rules)


def get_preprocessed_table_func(
    adj_table_name: str,
    version: Optional[Union[int, str, None]] = None
) -> callable:
    """Returns the corresponding preprocessing function for a given table name.

    Parameters
    ----------
    adj_table_name : str
        The sub-table to be joint to the main left one.
    version : Optional[Union[int, st, None]]
        The version of code to use: 1, 2, 3, or None (which defaults to 3).
        1 refers to the original version, 2 to the enhanced original version,
        and None or 3 refers to our fully derived version.
        TODO : à revoir : "baseline_v1", "baseline_v2", "freestyle_A", etc
    """
    from home_credit import old_kernel_v1, old_kernel_v2
    if version == 1:
        name_func_map = {
            "application": old_kernel_v1.application_train_test_v1,
            "bureau": old_kernel_v1.bureau_and_balance_v1,
            "previous_application": old_kernel_v1.previous_application_v1,
            "pos_cash_balance": old_kernel_v1.pos_cash_balance_v1,
            "credit_card_balance": old_kernel_v1.credit_card_balance_v1,
            "installments_payments": old_kernel_v1.installments_payments_v1
        }
    elif version == 2:
        name_func_map = {
            "application": old_kernel_v2.application_train_test_v2,
            "bureau": old_kernel_v2.bureau_and_balance_v2,
            "previous_application": old_kernel_v2.previous_application_v2,
            "pos_cash_balance": old_kernel_v2.pos_cash_balance_v2,
            "credit_card_balance": old_kernel_v2.credit_card_balance_v2,
            "installments_payments": old_kernel_v2.installments_payments_v2
        }
    else:  # if version == "freestyle_A"  
        name_func_map = {
            "application": preprocessed_application,
            "bureau": preprocessed_bureau_and_balance,
            "previous_application": preprocessed_previous_application,
            "pos_cash_balance": preprocessed_pos_cash_balance,
            "credit_card_balance": preprocessed_credit_card_balance,
            "installments_payments": preprocessed_installments_payments
        }
    return name_func_map.get(adj_table_name)


def left_join_grouped_table(
    data: pd.DataFrame,
    adj_table_name: str,
    nrows: int,
    version: Optional[Union[int, None]] = None,
    verbosity: Optional[Union[int, None]] = None
) -> pd.DataFrame:
    """
    Left join `data` with a preprocessed table.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to join with the preprocessed table.
    adj_table_name : str
        The name of the preprocessed table to join with.
    nrows : int
        The number of rows to read from the preprocessed table.
    version : Optional[Union[int, None]]
        The version of code to use: 1, 2, 3, or None (which defaults to 3).
        1 refers to the original version, 2 to the enhanced original version,
        and None or 3 refers to our fully derived version.
    verbosity : int, optional
        The level of verbosity. Defaults to None, which corresponds to 0 (quiet).
        If `debug` is True, then `verbosity` is set to at least 1.

    Returns
    -------
    pd.DataFrame
        The joined DataFrame.
    """
    inputs = {"data": data, "adj_table_name": adj_table_name, "nrows": nrows}
    outputs = {}  # Added by exec_tracking or others during execution
    # TODO : ctx mgr : mesure de la mémoire
    with exec_tracking(
        title=f"Preprocess `{adj_table_name}`",
        context=f"Version {version}",
        inputs=inputs,
        outputs=outputs,
        verbosity=verbosity
    ):
        proc_adj_table = get_preprocessed_table_func(adj_table_name, version)
        adj_table = proc_adj_table(nrows)
        outputs["adj_table"] = adj_table
        # data = pd.merge(data, adj_table, how="left", on="SK_ID_CURR")
        data = data.join(adj_table, how="left", on="SK_ID_CURR")
        outputs["updated_data"] = data
        # free(adj_table)
        return data


def main_preprocessing(
    nrows: Optional[Union[int, None]] = None,
    version: Optional[Union[int, None]] = None,
    verbosity: Optional[Union[int, None]] = None,
):
    """
    Preprocess the data to produce the input for the modeling.

    Parameters
    ----------
    nrows : int, optional
        The number of rows to read from the preprocessed table.
        If None, all rows are read.
    version : int, optional
        The version of code to use: 1, 2, 3, or None (which defaults to 3).
        1 refers to the original version, 2 to the enhanced original version,
        and None or 3 refers to our fully derived version.
    verbosity : int, optional
        The level of verbosity. Defaults to None, which corresponds to 0 (quiet).
        If `debug` is True, then `verbosity` is set to at least 1.

    Returns
    -------
    """
    params = nrows, version, verbosity
    # def data_check(msg, data):
    #     print(msg)
    #     data_objects = data.select_dtypes(include="object")
    #     display(data_objects.columns)

    data = get_preprocessed_table_func("application")(*params)
    # data_check("app", data)
    data = left_join_grouped_table(data, "bureau", *params)
    # data_check("app+bur", data)
    data = left_join_grouped_table(data, "previous_application", *params)
    # data_check("app+bur+prv", data)
    data = left_join_grouped_table(data, "pos_cash_balance", *params)
    # data_check("app+bur+prv+pcb", data)
    data = left_join_grouped_table(data, "credit_card_balance", *params)
    # data_check("app+bur+prv+pcb+ccb", data)
    data = left_join_grouped_table(data, "installments_payments", *params)
    # data_check("app+bur+prv+pcb+ccb+ip", data)

    """
    The successive joins may have introduced NAs in the hot encoded columns.
    Therefore, we proceed to a new pass of hot encoding on the object dtype columns,
    those that have been affected by these joins.
    """
    data, cats = one_hot_encoder(data)
    if verbosity > 2:
        print("Re-hot-encoded cats:")
        display(cats)

    # Hot encoding can introduce labels that are not utf-8
    safe_utf8_column_names(data)

    return data


def get_opt_lgbm_classifier(verbosity: int = 0) -> lgbm.LGBMClassifier:
    """
    Return a LightGBM classifier with optimal parameters found
    by Bayesian optimization.
    """
    # Parameters from Tilii kernel:
    # https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code

    # Ignore warnings related to verbose and early_stopping_rounds
    warnings.filterwarnings(
        "ignore", message=".*'verbose' argument is deprecated.*",
        category=UserWarning
    )
    warnings.filterwarnings(
        "ignore", message=".*'early_stopping_rounds' argument is deprecated.*",
        category=UserWarning
    )

    # Define the model parameters
    model_params = {
        # nthread=4,
        "n_jobs": 4,
        "n_estimators": 10_000,
        "learning_rate": .02,
        "num_leaves": 34,
        "colsample_bytree": .9497036,
        "subsample": .8715623,
        "max_depth": 8,
        "reg_alpha": .041545473,
        "reg_lambda": .0735294,
        "min_split_gain": .0222415,
        "min_child_weight": 39.3259775,
        # "silent=-1,
        # "verbose": -1,   # deprecated, replaced by `callback`
        # "early_stopping_rounds": None,
        # Use LGBMCallback for verbosity
        "callbacks": [
            lgbm.callbacks.LGBMCallback(logging_frequency=verbosity)
        ] if verbosity > 0 else None,
        # "stopping_rounds": 15,  # [10, 20]
        # See : https://lightgbm.readthedocs.io/en/latest/Parameters.html
    }

    return lgbm.LGBMClassifier(**model_params)


# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(
    data: pd.DataFrame,
    nfolds: int,
    stratified: bool = False,
    debug: bool = False,
    sms_filename: str = "submission_kernel.csv",
    # force_gc: bool = False
) -> pd.DataFrame:
    """
    Train a LightGBM Gradient Boosting Decision Tree model using KFold or
    Stratified KFold cross-validation.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the features and target variable.
    nfolds : int
        The number of folds for cross-validation.
    stratified : bool; optional
        Whether to use StratifiedKFold cross-validation.
        Defaults to False.
    debug : bool, optional
        Whether to print debugging information.
        Defaults to False.
    sms_filename : str, optional
        The filename of the submission file to write.
        Defaults to "submission_kernel.csv"
    # force_gc : bool, optional
    #     Whether to force garbage collection as soon as an object is no longer
    #     needed. Defaults to False.

    Returns
    -------
        A DataFrame containing the feature importances.
    """
    # Exclude non-feature columns from training and test features
    not_feat_names = ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
    feat_names = data.columns.difference(not_feat_names)
    X = data[feat_names]
    y = data.TARGET
    id_curr = data.SK_ID_CURR

    # Split the target variable and the features for training and test data
    X_train = X[y > -1].copy()
    y_train = y[y > -1].copy()
    X_test = X[y == -1].copy()
    id_test = id_curr[y == -1].copy()
    # y_test = y[y == -1].copy() aucun intérêt

    # Print the shape of the training and test data
    print(f"Starting LightGBM. Train shape: {X_train.shape}, test shape: {X_test.shape}")

    # Create the cross-validation model
    fold_params = {"n_splits": nfolds, "shuffle": True, "random_state": 42}
    folds = (StratifiedKFold if stratified else KFold)(**fold_params)

    # Create arrays and dataframes to store results
    # `oof_preds` will store the out-of-fold predictions for the training data
    # `sms_preds` will store the submission predictions for the test data 
    oof_preds = np.zeros(X_train.shape[0])
    sms_preds = np.zeros(X_test.shape[0])

    # Iterate through the folds
    fold_imps = []
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
        # Split the training and validation data using the current fold
        X_y_split = lambda idx: (X_train.iloc[idx], y_train.iloc[idx])
        X_y_train = X_y_split(train_idx)
        X_y_valid = X_y_split(valid_idx)
        X_valid, y_valid = X_y_valid

        # Train the LightGBM model using the training and validation data
        clf = get_opt_lgbm_classifier()
        clf.fit(
            *X_y_train,
            eval_set=[X_y_train, X_y_valid],
            eval_metric="auc", verbose=200, early_stopping_rounds=200
        )

        # Make predictions on the validation and test data using the trained model
        oof_preds[valid_idx] = clf.predict_proba(
            X_valid,
            num_iteration=clf.best_iteration_
        )[:, 1]

        # Aggregate the submission predictions (test predictions) for the current fold
        sms_preds += clf.predict_proba(
            X_test,
            num_iteration=clf.best_iteration_
        )[:, 1] / folds.n_splits

        # Get the feature importances for the current fold
        fold_imp = pd.DataFrame({
            "feature": feat_names,
            "importance": clf.feature_importances_,
            "fold": n_fold+1
        })

        # Concatenate the feature importances across all folds
        fold_imps.append(fold_imp)

        # Print the AUC for the current fold
        # Avoid ValueError:
        #   Only one class present in y_true. ROC AUC score is not defined in that case.
        if y_valid.nunique() > 1:
            kv(2,
               f"Fold {n_fold+1:2d} AUC",
               f"{roc_auc_score(y_valid, oof_preds[valid_idx]):.6f}"
            )
        else:
            print("Only one class present in `y_valid`. "
                  "ROC AUC score is not defined in that case.")

    # Print the overall train AUC
    if y_train.nunique() > 1:
        kv(2, "Full AUC score", f"{roc_auc_score(y_train, oof_preds):.6f}")
    else:
        print("Only one class present in `y_train`. "
              "ROC AUC score is not defined in that case.")

    # Write the submission file
    if not debug:
        sms_data = id_test.reset_index()
        sms_data["TARGET"] = sms_preds
        save_submission(sms_data, sms_filename)

    # Concatenate the feature importances across all folds
    feat_imp = pd.concat(fold_imps, axis=0)

    # Display the feature importance
    display_importances(feat_imp)

    # Force the garbage collection : the author must be a Java or C++ programmer
    # if force_gc:
    #     free(clf, X_train, X_test, y_train, y_test)

    # And return it
    return feat_imp


def safe_utf8_column_names(data: pd.DataFrame) -> None:
    """
    Sanitize column names of a pandas DataFrame to remove any special
    characters that may cause issues when used in other libraries,
    such as LightGBM.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame whose column names need to be sanitized.

    Returns
    -------
    None
        In-place.

    Note
    ----
    This function helps avoid the LightGBMError:
    "Do not support special JSON characters in feature name"
    that can occur when hot encoding introduces object values in column labels.
    """
    # Define regex pattern to match any character that is not a letter, digit or underscore
    pat = re.compile(r"[^\w]+")
    
    # Sanitize column names using the regex pattern, replacing any special characters with underscores
    clean_cols = [re.sub(pat, "_", col.strip()) for col in data.columns]
    
    # Check if any column names are empty or duplicated, and if so, assign them a generic name
    for i, col in enumerate(clean_cols):
        if not col:
            clean_cols[i] = f"col_{i}"
        if clean_cols[:i].count(col):
            clean_cols[i] = f"{col}_{clean_cols[:i].count(col)}"
    
    # Rename columns with the clean names
    data.rename(columns=dict(zip(data.columns, clean_cols)), inplace=True)


def main(
    nrows: Optional[Union[int, None]] = None,
    version: Optional[Union[int, None]] = None,
    verbosity: Optional[Union[int, None]] = None,
    sms_filename: str = "submission_kernel.csv",
    debug: Optional[Union[bool, None]] = False
) -> None:
    """
    Preprocess data and train a LightGBM model.

    Parameters
    ----------
    nrows : int, optional
        The number of rows to read from the preprocessed table.
        If None, all rows are read.
    version : int, optional
        The version of code to use: 1, 2, 3, or None (which defaults to 3).
        1 refers to the original version, 2 to the enhanced original version,
        and None or 3 refers to our fully derived version.
    verbosity : int, optional
        The level of verbosity. Defaults to None, which corresponds to 0 (quiet).
        If `debug` is True, then `verbosity` is set to at least 1.
    sms_filename : str, optional
        The filename of the submission file to write.
        Defaults to "submission_kernel.csv"
    debug : bool, optional
        If True, use only a subset of the data and fewer iterations.
    
    """
    nrows = 10_000 if debug and nrows is None else None
    verbosity = 0 if verbosity is None else verbosity
    verbosity = max(1, verbosity) if debug else verbosity
    params = nrows, version, verbosity

    data = main_preprocessing(*params)

    kfold_lightgbm_inputs = {
        "data": data,
        "nfolds": 10,
        "stratified": False,
        "debug": debug,
        "sms_filename": sms_filename
    }
    with exec_tracking(
        title="Run LightGBM with kfold",
        context=f"Version {version}",
        inputs=kfold_lightgbm_inputs,
        outputs={},
        verbosity=verbosity
    ):
        _ = kfold_lightgbm(**kfold_lightgbm_inputs)


if __name__ == "__main__":
    sms_filename = "submission_kernel.csv"
    with exec_tracking(
        title="Full model run",
        context="__main__",
        inputs={"sms_filename": sms_filename},
        outputs={},
        verbosity=1
    ):
        main(sms_filename=sms_filename)
