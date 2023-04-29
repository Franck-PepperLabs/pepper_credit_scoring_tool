# Inspired bu and derived from :
# https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features
# kaggle kernels output jsaguiar/lightgbm-with-simple-features -p /path/to/dest

# Common Python foundations imports
from typing import Optional, Union, List, Tuple
import re
import gc
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

""" Never do this, it's a very bad habit!
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
"""

# Home imports
from pepper.debug import tx, kv, tl, stl, sstl, this_f_name
from pepper.utils import save_and_show, pretty_timedelta_str
from pepper.pd_utils import align_df2_on_df1  # For `load_IP_table` in `silly_mode`

from home_credit.utils import get_table
from home_credit.feat_eng import nullify_365243
from home_credit.lightgbm_kernel import one_hot_encoder  # For compatibility check


def free(*objs: object) -> None:
    """Frees memory used by objects and garbage collect.

    Parameters
    ----------
    *objs : object
        The objects to free memory for.

    Returns
    -------
    None

    """
    for o in objs:
        del o
    gc.collect()


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
    """Loads a table from cache and sample rows if `nrows` is specified.

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
    """Get a list of categorical variables in a DataFrame.

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
    else:
        if isinstance(dtype, (list, tuple, np.ndarray, pd.Index, pd.Series)):
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
    """One-hot encode all categorical columns in the DataFrame.

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
    """One-hot encodes categorical features in the input DataFrame and returns
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
        Whether or not to drop the first column to avoid multicollinearity.
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


def add_derived_features(data: pd.DataFrame, expr: str) -> None:
    """Adds new derived features to a given DataFrame by evaluating an
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
    data.eval(expr, inplace=True, engine="numexpr")


def flatten_and_prefix_columns(data: pd.DataFrame, prefix: str = None) -> None:
    """Flattens MultiIndex columns and adds a prefix to all columns in the
    given DataFrame.
    
    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame whose columns will be flattened and prefixed.
    prefix : str, optional
        The string prefix to add to each column name. Defaults to None.

    """
    # Generate new column names by concatenating prefix, level 0, and level 1
    if prefix is None or prefix == "":
        prefix = ""
    else:
        prefix += "_"
    new_columns = [f"{prefix}{c[0]}_{c[1].upper()}" for c in data.columns]
    # Assign new column names to DataFrame columns
    data.columns = pd.Index(new_columns)



def display_importances(feat_imp: pd.DataFrame, title: str = None) -> None:
    """Displays a bar plot of feature importance.

    Parameters
    ----------
    feat_imp : pd.DataFrame)
        A dataframe with feature importance information.
    title : str, optional
        The title of the plot. Defaults to None.

    Returns
    -------
        None
    """
    # Get the top 40 features by importance
    cols = (
        feat_imp[["feature", "importance"]]
        .groupby("feature").mean()
        .sort_values(by="importance", ascending=False)
        [:40].index
    )

    # Sort the features by importance and get the best ones
    best_features = (
        feat_imp.loc[feat_imp.feature.isin(cols)]
        .sort_values(by="importance", ascending=False)
    )

    # Create the plot
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features)
    plt.title('LightGBM Features (avg over folds)')

    # Adjust the spacing between the subplots and save/show the figure
    plt.tight_layout()
    title = "default" if title is None else title
    save_and_show(f"feature_importace_{title.lower()}", sub_dir="feat_imp")


""" V1 vs V2
"""

def load_A_table_v1(nrows=None):
    """DEPRECATED use `load_A_table_v2` instead"""
    # Read data and merge
    data = pd.read_csv('../../dataset/csv/application_train.csv', nrows=nrows)
    test_data = pd.read_csv('../../dataset/csv/application_test.csv', nrows=nrows)
    # print(f"Train samples: {len(df)}, test samples: {len(test_df)}")
    # NB: `append` doesn't exist in current Pandas 2.0, replaced by `concat`
    #     `append` has been deprecated since version 1.3.0 of Pandas (June 2021)
    # NB2: A reset_index() statement in older code (< 1.3.0) is equivalent to reset_index(drop=True)
    # in modern code, due to the change in the default value of the drop parameter.
    # data = data.append(test_data).reset_index()
    data = pd.concat([data, test_data], axis=0)
    data = data.reset_index(drop=True)
    free(test_data)
    return data


def load_A_table_v2(nrows=None, silly_mode=False):
    """DEPRECATED use `load_data` instead"""
    data = load_data("application", nrows)
    if silly_mode:   # The art of going with the flow
        data.TARGET = data.TARGET.astype(object).replace(-1, np.nan)
        pk_name = "SK_ID_CURR"
        data_v1 = load_A_table_v1(nrows)
        data = align_df2_on_df1(pk_name, data_v1, data)
    return data


def load_B_tables_v1(nrows=None):
    """DEPRECATED use `load_B_table_v2` instead"""
    # Note that selecting the first `nrows` rows respectively from the two tables 
    # does not guarantee that their `SK_ID_BUREAU` will have a non-empty intersection,
    # so joining them with `merge` or `join` may result in significant loss.
    data = pd.read_csv('../../dataset/csv/bureau.csv', nrows=nrows)
    adj_data = pd.read_csv('../../dataset/csv/bureau_balance.csv', nrows=nrows)
    return data, adj_data


def load_B_tables_v2(nrows=None):
    """DEPRECATED use `load_data` instead"""
    """ If the `nrows` argument is not `None`:
    1/ First, we perform a random sampling of `bureau`
    2/ Then, we extract the rows from `bureau_balance` that
    share the same `SK_ID_BUREAU` as the `bureau` sample."""
    data = load_data("bureau", nrows)
    adj_data = load_data("bureau_balance")
    if nrows is not None:
        adj_data = adj_data[adj_data.SK_ID_BUREAU.isin(data.SK_ID_BUREAU)]
    return data, adj_data


def load_PA_table_v1(nrows=None):
    """DEPRECATED use `load_PA_table_v2` instead"""
    return pd.read_csv('../../dataset/csv/previous_application.csv', nrows=nrows)


def load_PA_table_v2(nrows=None):
    """DEPRECATED use `load_data` instead"""
    return load_data("previous_application", nrows)


def load_PCB_table_v1(nrows=None):
    """DEPRECATED use `load_PCB_table_v2` instead"""
    return pd.read_csv('../../dataset/csv/POS_CASH_balance.csv', nrows=nrows)


def load_PCB_table_v2(nrows=None):
    """DEPRECATED use `load_data` instead"""
    return load_data("pos_cash_balance", nrows)


def load_CCB_table_v1(nrows=None):
    """DEPRECATED use `load_CCB_table_v2` instead"""
    return pd.read_csv('../../dataset/csv/credit_card_balance.csv', nrows=nrows)


def load_CCB_table_v2(nrows=None):
    """DEPRECATED use `load_data` instead"""
    return load_data("credit_card_balance", nrows)


def load_IP_table_v1(nrows=None):
    return pd.read_csv('../../dataset/csv/installments_payments.csv', nrows=nrows)


def load_IP_table_v2(nrows=None):
    return load_data("installments_payments", nrows)


def clean_A_cats_v1(data):
    """DEPRECATED use `clean_A_cats_v2` instead"""
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    # NB > copy() was added to avoid subsequent errors or warnings caused by working on a view
    return data[data['CODE_GENDER'] != 'XNA'].copy()


def clean_A_cats_v2(data):
    """DEPRECATED use `clean_A_table` instead"""
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    # NB > Here, there's no need for a copy, as we are working directly in-place
    data.drop(index=data.index[data.CODE_GENDER == "XNA"], inplace=True)
    # For compatibility with V1, but not necessary, it's in-place
    return data


def clean_A_nums_v1(data):
    """DEPRECATED use `clean_A_nums_v2` instead"""
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)


def clean_A_nums_v2(data):
    # NaN values for DAYS_*: 365.243 -> nan
    nullify_365243(data.DAYS_EMPLOYED)


def clean_PA_nums_v1(data):
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    data['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    data['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    data['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    data['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    data['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)


def clean_PA_nums_v2(data):
    # NaN values for DAYS_*: 365.243 -> nan
    cols = data.columns
    days_cols = cols[cols.str.match("DAYS_")]
    # NB : don't try the call nullify_365243(data[days_cols])
    # It would produce the SettingWithCopyWarning:
    #   A value is trying to be set on a copy of a slice from a DataFrame
    [nullify_365243(data[col]) for col in days_cols]


def clean_A_data(data):
    pass


def clean_B_data(data):
    pass


def clean_PA_data(data):
    pass


def clean_PCB_data(data):
    pass


def clean_CCB_data(data):
    pass


def clean_IP_data(data):
    pass


def encode_A_bin_cats_v1(data):    
    """DEPRECATED use `clean_IP_cats_v2` instead"""
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        # Unused `uniques` has been replaced by `_`
        data[bin_feature], _ = pd.factorize(data[bin_feature])


def encode_A_bin_cats_v2(data):    
    """DEPRECATED use `hot_encode_cats` instead"""
    # Categorical features with Binary encode (0 or 1; two categories)
    bin_vars = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    for bin_var in bin_vars:
        data[bin_var] = data[bin_var].astype("category").cat.codes
    # For compatibility check
    data.CODE_GENDER = 1 - data.CODE_GENDER
    data.FLAG_OWN_REALTY = 1 - data.FLAG_OWN_REALTY


def group_BB_by_bur_and_join_to_B_v1(data, adj_data, adj_catvar_names):
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in adj_catvar_names:
        bb_aggregations[col] = ['mean']
    data_adj_agg = adj_data.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    data_adj_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in data_adj_agg.columns.tolist()])
    data = data.join(data_adj_agg, how='left', on='SK_ID_BUREAU')
    data.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    free(adj_data, data_adj_agg)
    return data


def group_BB_by_bur_and_join_to_B_v2(data, adj_data, adj_catvar_names):
    # Bureau balance: Perform aggregations and merge with bureau.csv
    agg_rules = {"MONTHS_BALANCE": ["min", "max", "size"]}
    agg_rules.update({col: ["mean"] for col in adj_catvar_names})
    data_adj_agg = adj_data.groupby("SK_ID_BUREAU").agg(agg_rules)
    flatten_and_prefix_columns(data_adj_agg)
    data = data.join(data_adj_agg, how="left", on="SK_ID_BUREAU")
    data.drop(["SK_ID_BUREAU"], axis=1, inplace= True)
    free(adj_data, data_adj_agg)
    return data


def group_B_by_curr_v1(data, catvar_names, adj_catvar_names):    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in catvar_names:
        cat_aggregations[cat] = ['mean']
    for cat in adj_catvar_names:
        cat_aggregations[cat + "_MEAN"] = ['mean']
    
    data_agg = data.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    data_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in data_agg.columns.tolist()])
    
    # Bureau: Active credits - using only numerical aggregations
    active = data[data.CREDIT_ACTIVE_Active == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    data_agg = data_agg.join(active_agg, how='left', on='SK_ID_CURR')
    free(active, active_agg)

    # Bureau: Closed credits - using only numerical aggregations
    closed = data[data.CREDIT_ACTIVE_Closed == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    data_agg = data_agg.join(closed_agg, how='left', on='SK_ID_CURR')

    # Free data
    free(closed, closed_agg, data)

    return data_agg


def group_B_by_curr_v2(data, catvar_names, adj_catvar_names):
    # Previous applications numeric features
    m_rule = ["mean"]
    # s_rule = ["sum"]
    mm_rule = ["max", "mean"]
    ms_rule = ["mean", "sum"]
    mms_rule = ["max", "mean", "sum"]
    mmm_rule = ["min", "max", "mean"]
    mmmv_rule = mmm_rule + ["var"]
    num_agg_rules = {
        "DAYS_CREDIT": mmmv_rule,
        "DAYS_CREDIT_ENDDATE": mmm_rule,
        "DAYS_CREDIT_UPDATE": m_rule,
        "CREDIT_DAY_OVERDUE": mm_rule,
        "AMT_CREDIT_MAX_OVERDUE": m_rule,
        "AMT_CREDIT_SUM": mms_rule,
        "AMT_CREDIT_SUM_DEBT": mms_rule,
        "AMT_CREDIT_SUM_OVERDUE": m_rule,
        "AMT_CREDIT_SUM_LIMIT": ms_rule,
        "AMT_ANNUITY": mm_rule,
        "CNT_CREDIT_PROLONG": ["sum"],
        "MONTHS_BALANCE_MIN": ["min"],
        "MONTHS_BALANCE_MAX": ["max"],
        "MONTHS_BALANCE_SIZE": ms_rule
    }

    # Bureau and bureau_balance categorical features
    cat_agg_rules = {c: m_rule for c in catvar_names}
    cat_agg_rules.update({c + "_MEAN": m_rule for c in adj_catvar_names})
    
    data_agg = data.groupby("SK_ID_CURR").agg({**num_agg_rules, **cat_agg_rules})
    flatten_and_prefix_columns(data_agg, "BURO")

    # Bureau: Active credits - using only numerical aggregations
    active = data[data.CREDIT_ACTIVE_Active == 1]
    active_agg = active.groupby("SK_ID_CURR").agg(num_agg_rules)
    flatten_and_prefix_columns(active_agg, "ACTIVE")
    data_agg = data_agg.join(active_agg, how="left", on="SK_ID_CURR")
    free(active, active_agg)

    # Bureau: Closed credits - using only numerical aggregations
    closed = data[data.CREDIT_ACTIVE_Closed == 1]
    closed_agg = closed.groupby("SK_ID_CURR").agg(num_agg_rules)
    flatten_and_prefix_columns(closed_agg, "CLOSED")
    data_agg = data_agg.join(closed_agg, how="left", on="SK_ID_CURR")

    # Free data
    free(closed, closed_agg, data)

    return data_agg


def group_PA_by_curr_v1(data, catvar_names):
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in catvar_names:
        cat_aggregations[cat] = ['mean']
    
    data_agg = data.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    data_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in data_agg.columns.tolist()])
    
    # Previous Applications: Approved Applications - only numerical features
    approved = data[data['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    data_agg = data_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    
    # Previous Applications: Refused Applications - only numerical features
    refused = data[data['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    data_agg = data_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    
    # Free data
    free(refused, refused_agg, approved, approved_agg, data)

    return data_agg


def group_PA_by_curr_v2(data, catvar_names):
    # Previous applications numeric features
    m_rule = ["mean"]
    ms_rule = ["mean", "sum"]
    mmm_rule = ["min", "max", "mean"]
    mmmv_rule = ["min", "max", "mean", "var"]
    num_agg_rules = {
        "AMT_ANNUITY": mmm_rule,
        "AMT_APPLICATION": mmm_rule,
        "AMT_CREDIT": mmm_rule,
        "APP_CREDIT_PERC": mmmv_rule,
        "AMT_DOWN_PAYMENT": mmm_rule,
        "AMT_GOODS_PRICE": mmm_rule,
        "HOUR_APPR_PROCESS_START": mmm_rule,
        "RATE_DOWN_PAYMENT": mmm_rule,
        "DAYS_DECISION": mmm_rule,
        "CNT_PAYMENT": ms_rule,
    }
    # Previous applications categorical features
    cat_agg_rules = {cat: m_rule for cat in catvar_names}   

    data_agg = data.groupby("SK_ID_CURR").agg({**num_agg_rules, **cat_agg_rules})
    flatten_and_prefix_columns(data_agg, "PREV")
    
    # TODO : Pourquoi séparer ? pour réduire aux moyennes locales ?
    # mais beaucoup de NA vont apparaître ?!

    # Previous Applications: Approved Applications - only numerical features
    approved = data[data.NAME_CONTRACT_STATUS_Approved == 1]
    approved_agg = approved.groupby("SK_ID_CURR").agg(num_agg_rules)
    flatten_and_prefix_columns(approved_agg, "APPROVED")
    data_agg = data_agg.join(approved_agg, how="left", on="SK_ID_CURR")
    
    # Previous Applications: Refused Applications - only numerical features
    refused = data[data.NAME_CONTRACT_STATUS_Refused == 1]
    refused_agg = refused.groupby("SK_ID_CURR").agg(num_agg_rules)
    flatten_and_prefix_columns(refused_agg, "REFUSED")
    data_agg = data_agg.join(refused_agg, how="left", on="SK_ID_CURR")
    
    # Free data
    free(refused, refused_agg, approved, approved_agg, data)

    return data_agg


def group_PCB_by_curr_v1(data, catvar_names):
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in catvar_names:
        aggregations[cat] = ['mean']
    
    data_agg = data.groupby('SK_ID_CURR').agg(aggregations)
    data_agg.columns = pd.Index([
        'POS_' + e[0] + "_" + e[1].upper()
        for e in data_agg.columns.tolist()
    ])    # Count pos cash accounts
    data_agg['POS_COUNT'] = data.groupby('SK_ID_CURR').size()

    # Free data
    free(data)

    return data_agg


def group_PCB_by_curr_v2(data, catvar_names):
    mm_rule = ["max", "mean"]
    mms_rule = mm_rule + ["size"]
    agg_rules = {
        "MONTHS_BALANCE": mms_rule,
        "SK_DPD": mm_rule,
        "SK_DPD_DEF": mm_rule
    }
    agg_rules.update({c: ["mean"] for c in catvar_names})
    gpby_curr = data.groupby("SK_ID_CURR")
    data_agg = gpby_curr.agg(agg_rules)
    flatten_and_prefix_columns(data_agg, "POS")
    
    # Count pos cash accounts
    data_agg["POS_COUNT"] = gpby_curr.size()

    # Free data
    free(data)

    return data_agg


def group_CCB_by_curr_v1(data, catvar_names):
    # General aggregations
    ## TODO : Pas à sa place, c'est du nettoyage : regrouper les clean
    data.drop(['SK_ID_PREV'], axis=1, inplace=True)
    data_agg = data.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    data_agg.columns = pd.Index([
        'CC_' + e[0] + "_" + e[1].upper()
        for e in data_agg.columns.tolist()
    ])
    
    # Count credit card lines
    data_agg['CC_COUNT'] = data.groupby('SK_ID_CURR').size()

    # Free data
    free(data)

    return data_agg


def group_CCB_by_curr_v2(data, catvar_names):
    ## TODO : Pas à sa place, c'est du nettoyage : regrouper les clean
    data.drop(['SK_ID_PREV'], axis=1, inplace=True)
    
    agg_rule = ["min", "max", "mean", "sum", "var"]
    gpby_curr = data.groupby('SK_ID_CURR')
    data_agg = gpby_curr.agg(agg_rule)
    flatten_and_prefix_columns(data_agg, "CC")

    # Count installments accounts
    data_agg["CC_COUNT"] = gpby_curr.size()
    
    # Free data
    free(data)

    return data_agg


def group_IP_by_curr_v1(data, catvar_names):
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in catvar_names:
        aggregations[cat] = ['mean']
    data_agg = data.groupby('SK_ID_CURR').agg(aggregations)
    data_agg.columns = pd.Index([
        'INSTAL_' + e[0] + "_" + e[1].upper()
        for e in data_agg.columns.tolist()
    ])
    # Count installments accounts
    data_agg['INSTAL_COUNT'] = data.groupby('SK_ID_CURR').size()

    # Free data
    free(data)
    
    return data_agg


def group_IP_by_curr_v2(data, catvar_names):
    # Features: Perform aggregations
    nu = ["nunique"]
    mms = ["max", "mean", "sum"]
    mmsv = mms + ["var"]
    mmms = ["min"] + mms
    agg_rules = {
        'NUM_INSTALMENT_VERSION': nu,
        'DPD': mms, 'DBD': mms,
        'PAYMENT_PERC': mmsv, 'PAYMENT_DIFF': mmsv,
        'AMT_INSTALMENT': mms, 'AMT_PAYMENT': mmms,
        'DAYS_ENTRY_PAYMENT': mms
    }
    agg_rules.update({cat: ["mean"] for cat in catvar_names})

    gpby_curr = data.groupby('SK_ID_CURR')
    data_agg = gpby_curr.agg(agg_rules)
    flatten_and_prefix_columns(data_agg, "INSTAL")

    data_agg["INSTAL_COUNT"] = gpby_curr.size()
    return data_agg


def add_A_derived_features_v1(data):
    """DEPRECATED use `add_A_derived_features` instead"""
    # Some simple new features (percentages)
    data['DAYS_EMPLOYED_PERC'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
    data['INCOME_CREDIT_PERC'] = data['AMT_INCOME_TOTAL'] / data['AMT_CREDIT']
    data['INCOME_PER_PERSON'] = data['AMT_INCOME_TOTAL'] / data['CNT_FAM_MEMBERS']
    data['ANNUITY_INCOME_PERC'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
    data['PAYMENT_RATE'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']


def add_A_derived_features(data):
    add_derived_features(
        data,
        """
        DAYS_EMPLOYED_PERC = DAYS_EMPLOYED / DAYS_BIRTH
        INCOME_CREDIT_PERC = AMT_INCOME_TOTAL / AMT_CREDIT
        INCOME_PER_PERSON = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS
        ANNUITY_INCOME_PERC = AMT_ANNUITY / AMT_INCOME_TOTAL
        PAYMENT_RATE = AMT_ANNUITY / AMT_CREDIT
        """
    )


def add_B_derived_features(data):
    pass


def add_PCB_derived_features(data):
    pass


def add_CCB_derived_features(data):
    pass


def add_PA_derived_features_v1(data):
    """DEPRECATED use `add_PA_derived_features` instead"""
    # Add feature: value ask / value received percentage
    data['APP_CREDIT_PERC'] = data['AMT_APPLICATION'] / data['AMT_CREDIT']


def add_PA_derived_features(data):
    add_derived_features(
        data,
        # Add feature: value ask / value received percentage
        """APP_CREDIT_PERC = AMT_APPLICATION / AMT_CREDIT"""
    )


def add_IP_derived_features_v1(data):
    """DEPRECATED use `add_IP_derived_features` instead"""
    # Percentage and difference paid in each installment (amount paid and installment value)
    data['PAYMENT_PERC'] = data['AMT_PAYMENT'] / data['AMT_INSTALMENT']
    data['PAYMENT_DIFF'] = data['AMT_INSTALMENT'] - data['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    data['DPD'] = data['DAYS_ENTRY_PAYMENT'] - data['DAYS_INSTALMENT']
    data['DBD'] = data['DAYS_INSTALMENT'] - data['DAYS_ENTRY_PAYMENT']
    data['DPD'] = data['DPD'].apply(lambda x: x if x > 0 else 0)
    data['DBD'] = data['DBD'].apply(lambda x: x if x > 0 else 0)


def add_IP_derived_features(data):
    add_derived_features(
        data,
        """
        PAYMENT_PERC = AMT_PAYMENT / AMT_INSTALMENT   # Percentage paid in each installment
        PAYMENT_DIFF = AMT_INSTALMENT - AMT_PAYMENT   # Difference paid in each installment
        DPD = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT    # Days past due (no negative values)
        DBD = DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT    # Days before due (no negative values)
        DPD = @where(DPD > 0, DPD, 0)
        DBD = @where(DBD > 0, DBD, 0)
        """
    )


def application_train_test_v1(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    data = load_A_table_v1(nrows)
    data = clean_A_cats_v1(data)
    encode_A_bin_cats_v1(data)
    data, _ = one_hot_encoder(data, nan_as_category)
    clean_A_nums_v1(data)
    add_A_derived_features_v1(data)
    return data


def application_train_test_v2(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    data = load_A_table_v2(nrows, silly_mode=True)
    clean_A_cats_v2(data)
    encode_A_bin_cats_v2(data)
    data, _ = hot_encode_cats(
        data,
        dummy_na=nan_as_category,
        discard_constants=False
    )
    clean_A_nums_v2(data)
    add_A_derived_features(data)
    return data


def preproc_application(
    nrows=None,
    dummy_na=True,
    drop_first=True
) -> pd.DataFrame:
    data = load_data("application", nrows)
    clean_A_data(data)
    data, _ = hot_encode_cats(data, dummy_na=dummy_na, drop_first=drop_first)
    add_A_derived_features(data)
    return data


def bureau_and_balance_v1(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    data, adj_data = load_B_tables_v1(nrows)
    data, catvars = one_hot_encoder(data, nan_as_category)
    adj_data, adj_catvars = one_hot_encoder(adj_data, nan_as_category)
    data = group_BB_by_bur_and_join_to_B_v1(data, adj_data, adj_catvars)
    return group_B_by_curr_v1(data, catvars, adj_catvars)


def bureau_and_balance_v2(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    data, adj_data = load_B_tables_v2(nrows)
    data, catvars = hot_encode_cats(data, nan_as_category)
    adj_data, adj_catvars = hot_encode_cats(adj_data, nan_as_category)
    data = group_BB_by_bur_and_join_to_B_v2(data, adj_data, adj_catvars)
    return group_B_by_curr_v2(data, catvars, adj_catvars)


def preproc_bureau(
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


def previous_application_v1(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    data = load_PA_table_v1(nrows)
    data, cats = one_hot_encoder(data, nan_as_category)
    clean_PA_nums_v1(data)
    add_PA_derived_features_v1(data)
    return group_PA_by_curr_v1(data, cats)


def previous_application_v2(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    data = load_PA_table_v2(nrows)
    data, cats = hot_encode_cats(
        data,
        dummy_na=nan_as_category,
        discard_constants=False
    )
    clean_PA_nums_v2(data)
    add_PA_derived_features(data)
    return group_PA_by_curr_v2(data, cats)


def preproc_previous_application(
    nrows=None,
    dummy_na=True,
    drop_first=True
) -> pd.DataFrame:
    data = load_data("previous_application", nrows)
    clean_PA_data(data)
    data, cats = hot_encode_cats(data, dummy_na=dummy_na, drop_first=drop_first)
    add_PA_derived_features(data)
    return group_PA_by_curr_v2(data, cats)


def pos_cash_balance_v1(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    data = load_PCB_table_v1(nrows)
    data, cats = one_hot_encoder(data, nan_as_category)
    return group_PCB_by_curr_v1(data, cats)


def pos_cash_balance_v2(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    data = load_PCB_table_v2(nrows)
    data, cats = hot_encode_cats(
        data,
        dummy_na=nan_as_category,
        discard_constants=False
    )
    return group_PCB_by_curr_v2(data, cats)


def preproc_pos_cash_balance(
    nrows=None,
    dummy_na=True,
    drop_first=True
) -> pd.DataFrame:
    data = load_data("pos_cash_balance", nrows)
    clean_PCB_data(data)
    data, cats = hot_encode_cats(data, dummy_na=dummy_na, drop_first=drop_first)
    add_PCB_derived_features(data)
    return group_PCB_by_curr_v2(data, cats)


def credit_card_balance_v1(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    data = load_CCB_table_v1(nrows)
    data, cats = one_hot_encoder(data, nan_as_category)
    return group_CCB_by_curr_v1(data, cats)


def credit_card_balance_v2(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    data = load_CCB_table_v2(nrows)
    data, cats = hot_encode_cats(
        data,
        dummy_na=nan_as_category,
        discard_constants=False
    )
    return group_CCB_by_curr_v2(data, cats)


def preproc_credit_card_balance(
    nrows=None,
    dummy_na=True,
    drop_first=True
) -> pd.DataFrame:
    data = load_data("credit_card_balance", nrows)
    clean_CCB_data(data)
    data, cats = hot_encode_cats(data, dummy_na=dummy_na, drop_first=drop_first)
    add_CCB_derived_features(data)
    return group_CCB_by_curr_v2(data, cats)


def installments_payments_v1(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    data = load_IP_table_v1(nrows)
    data, cats = one_hot_encoder(data, nan_as_category)
    add_IP_derived_features_v1(data)
    return group_IP_by_curr_v1(data, cats)


def installments_payments_v2(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    data = load_IP_table_v2(nrows)
    data, cats = hot_encode_cats(
        data,
        dummy_na=nan_as_category,
        discard_constants=False
    )
    add_IP_derived_features(data)
    return group_IP_by_curr_v2(data, cats)


def preproc_installments_payments(
    nrows=None,
    dummy_na=True,
    drop_first=True
) -> pd.DataFrame:
    data = load_data("installments_payments", nrows)
    clean_IP_data(data)
    data, cats = hot_encode_cats(data, dummy_na=dummy_na, drop_first=drop_first)
    add_IP_derived_features(data)
    return group_IP_by_curr_v2(data, cats)


def get_preproc_table_func(
    adj_table_name: str,
    version: Optional[Union[int, None]] = None
) -> callable:
    """Returns the corresponding preprocessing function for a given table name.

    Parameters
    ----------
    adj_table_name : str
        The subtable to be joint to the main left one.
    version : Optional[Union[int, None]]
        The version of code to use: 1, 2, 3, or None (which defaults to 3).
        1 refers to the original version, 2 to the enhanced original version,
        and None or 3 refers to our fully derived version.
    """
    if version == 1:
        name_func_map = {
            "application": application_train_test_v1,
            "bureau": bureau_and_balance_v1,
            "previous_application": previous_application_v1,
            "pos_cash_balance": pos_cash_balance_v1,
            "credit_card_balance": credit_card_balance_v1,
            "installments_payments": installments_payments_v1
        }
    elif version == 2:
        name_func_map = {
            "application": application_train_test_v2,
            "bureau": bureau_and_balance_v2,
            "previous_application": previous_application_v2,
            "pos_cash_balance": pos_cash_balance_v2,
            "credit_card_balance": credit_card_balance_v2,
            "installments_payments": installments_payments_v2
        }
    else:    
        name_func_map = {
            "application": preproc_application,
            "bureau": preproc_bureau,
            "previous_application": preproc_previous_application,
            "pos_cash_balance": preproc_pos_cash_balance,
            "credit_card_balance": preproc_credit_card_balance,
            "installments_payments": preproc_installments_payments
        }
    return name_func_map.get(adj_table_name)


def left_join_grouped_table(
    data: pd.DataFrame,
    adj_table_name: str,
    nrows: int,
    version: Optional[Union[int, None]] = None,
    verbosity: Optional[Union[int, None]] = None
) -> pd.DataFrame:
    """Left joins `data` with a preprocessed table.

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
        proc_adj_table = get_preproc_table_func(adj_table_name, version)
        adj_table = proc_adj_table(nrows)
        outputs["adj_table"] = adj_table
        # data = pd.merge(data, adj_table, how="left", on="SK_ID_CURR")
        data = data.join(adj_table, how="left", on="SK_ID_CURR")
        outputs["updated_data"] = data
        # free(adj_table)
        return data


def main_preproc(
    nrows: Optional[Union[int, None]] = None,
    version: Optional[Union[int, None]] = None,
    verbosity: Optional[Union[int, None]] = None,
):
    """Preprocesses the data to produce the input for the modeling.

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

    data = get_preproc_table_func("application")(*params)
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


def get_opt_lgbm_classifier() -> lgbm.LGBMClassifier:
    """Returns a LightGBM classifier with optimal parameters found
    by Bayesian optimization."""
    # Parameters from Tilii kernel:
    # https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
    stopping_rounds = 15  # [10, 20]
    return lgbm.LGBMClassifier(
        nthread=4,
        n_estimators=10000,
        learning_rate=0.02,
        num_leaves=34,
        colsample_bytree=0.9497036,
        subsample=0.8715623,
        max_depth=8,
        reg_alpha=0.041545473,
        reg_lambda=0.0735294,
        min_split_gain=0.0222415,
        min_child_weight=39.3259775,
        silent=-1,
        verbose=-1,
        # See : https://lightgbm.readthedocs.io/en/latest/Parameters.html
    )


# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(
    data: pd.DataFrame,
    nfolds: int,
    stratified: bool = False,
    debug: bool = False,
    sms_filename: str = "submission_kernel.csv"
) -> pd.DataFrame:
    """Trains a LightGBM Gradient Boosting Decision Tree model using KFold or
    Stratified KFold cross-validation.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the features and target variable.
    nfolds : int
        The number of folds for cross-validation.
    stratified : bool; optional
        Whether to use StratifiedKFold cross-validation. Defaults to False.
    debug : bool, optional
        Whether to print debugging information. Defaults to False.
    sms_filename : str, optional
        The filename of the submission file to write.
        Defaults to "submission_kernel.csv"

    Returns
    -------
        A DataFrame containing the feature importances.
    """
    # Exclude some non-feature columns from train and test data
    not_feat_names = ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
    feat_names = data.columns.difference(not_feat_names)
    X = data[feat_names]
    y = data.TARGET

    # Get the target variable and the features for training and test data
    X_train = X[y > -1].copy()
    X_test = X[y == -1].copy()
    y_train = y[y > -1].copy()
    y_test = y[y == -1].copy()

    # Print the shape of the training and test data
    print(f"Starting LightGBM. Train shape: {X_train.shape}, test shape: {X_test.shape}")
    
    # Cross validation model
    fold_params = {"n_splits": nfolds, "shuffle": True, "random_state": 42}
    folds = (StratifiedKFold if stratified else KFold)(*fold_params)
    
    # Create arrays and dataframes to store results
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

        # Aggregate the submission predictions for the current fold
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
        print(
            f"Fold {n_fold+1:2d} - "
            f"AUC : {roc_auc_score(y_valid, oof_preds[valid_idx]):.6f}"
        )

        # Free up memory
        free(clf, *X_y_train, *X_y_valid)

    print(f"Full AUC score {roc_auc_score(y_test, oof_preds):.6f}")
    
    # Write submission file and plot feature importance
    if not debug:
        y_test = sms_preds
        X_test[["SK_ID_CURR", "TARGET"]].to_csv(sms_filename, index=False)

    # Concatenate the feature importances across all folds
    feat_imp = pd.concat(fold_imps, axis=0)

    # Display the feature importances
    display_importances(feat_imp)

    # And return it
    return feat_imp


def safe_utf8_column_names(data: pd.DataFrame) -> None:
    """Sanitizes column names of a pandas DataFrame to remove any special
    characters that may cause issues when used in other libraries, such as
    LightGBM.

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
    """Preprocesses data and train a LightGBM model.

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

    data = main_preproc(data, *params)

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
        context=f"__main__",
        inputs={"sms_filename": sms_filename},
        outputs={},
        verbosity=1
    ):
        main(sms_filename=sms_filename)
