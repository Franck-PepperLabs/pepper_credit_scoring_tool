import numpy as np
import pandas as pd

from home_credit.kernel import (
    load_data,
    flatten_and_prefix_columns,
    add_derived_features,
    hot_encode_cats
)

from home_credit.old_kernel_v1 import load_A_table_v1, free

from pepper.pd_utils import align_df2_on_df1
from home_credit.feat_eng import nullify_365243


""" Loading
"""


def load_A_table_v2(nrows=None, silly_mode=False):
    """DEPRECATED use `load_data` instead.
    Load `application` table."""
    data = load_data("application", nrows)
    if silly_mode:   # The art of going with the flow
        data.TARGET = data.TARGET.astype(object).replace(-1, np.nan)
        pk_name = "SK_ID_CURR"
        data_v1 = load_A_table_v1(nrows)
        data = align_df2_on_df1(pk_name, data_v1, data)
    return data


def load_B_tables_v2(nrows=None):
    """DEPRECATED use `load_data` instead.
    Load `bureau` and `bureau_balance` tables.
    If the `nrows` argument is not `None`:
    1/ First, we perform a random sampling of `bureau`
    2/ Then, we extract the rows from `bureau_balance` that
    share the same `SK_ID_BUREAU` as the `bureau` sample."""
    data = load_data("bureau", nrows)
    adj_data = load_data("bureau_balance")
    if nrows is not None:
        adj_data = adj_data[adj_data.SK_ID_BUREAU.isin(data.SK_ID_BUREAU)]
    return data, adj_data


def load_PA_table_v2(nrows=None):
    """DEPRECATED use `load_data` instead.
    Load `previous_application` table."""
    return load_data("previous_application", nrows)


def load_PCB_table_v2(nrows=None):
    """DEPRECATED use `load_data` instead.
    Load `pos_cash_balance` table."""
    return load_data("pos_cash_balance", nrows)


def load_CCB_table_v2(nrows=None):
    """DEPRECATED use `load_data` instead.
    Load `credit_card_balance` table."""
    return load_data("credit_card_balance", nrows)


def load_IP_table_v2(nrows=None):
    """DEPRECATED use `load_data` instead.
    Load `installments_payments` table."""
    return load_data("installments_payments", nrows)


""" Cleaning
"""

def clean_A_cats_v2(data):
    """DEPRECATED use `clean_A_table` instead"""
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    # NB > Here, there's no need for a copy, as we are working directly in-place
    data.drop(index=data.index[data.CODE_GENDER == "XNA"], inplace=True)
    # For compatibility with V1, but not necessary, it's in-place
    return data


def clean_A_nums_v2(data):
    # NaN values for DAYS_*: 365.243 -> nan
    nullify_365243(data.DAYS_EMPLOYED)


def clean_PA_nums_v2(data):
    # NaN values for DAYS_*: 365.243 -> nan
    cols = data.columns
    days_cols = cols[cols.str.match("DAYS_")]
    # NB : don't try the call nullify_365243(data[days_cols])
    # It would produce the SettingWithCopyWarning:
    #   A value is trying to be set on a copy of a slice from a DataFrame
    [nullify_365243(data[col]) for col in days_cols]


def clean_A_data_v2(data):
    pass


def clean_B_data_v2(data):
    pass


def clean_PA_data_v2(data):
    pass


def clean_PCB_data_v2(data):
    pass


def clean_CCB_data_v2(data):
    pass


def clean_IP_data_v2(data):
    pass


""" Encoding
"""


def encode_A_bin_cats_v2(data):    
    """DEPRECATED use `hot_encode_cats` instead"""
    # Categorical features with Binary encode (0 or 1; two categories)
    bin_vars = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    for bin_var in bin_vars:
        data[bin_var] = data[bin_var].astype("category").cat.codes
    # For compatibility check
    data.CODE_GENDER = 1 - data.CODE_GENDER
    data.FLAG_OWN_REALTY = 1 - data.FLAG_OWN_REALTY


""" Grouping
"""

def group_BB_by_bur_and_join_to_B_v2(data, adj_data, adj_catvar_names):
    agg_rules = {"MONTHS_BALANCE": ["min", "max", "size"]} | {
        col: ["mean"] for col in adj_catvar_names
    }
    data_adj_agg = adj_data.groupby("SK_ID_BUREAU").agg(agg_rules)
    flatten_and_prefix_columns(data_adj_agg)
    data = data.join(data_adj_agg, how="left", on="SK_ID_BUREAU")
    data.drop(["SK_ID_BUREAU"], axis=1, inplace=True)
    free(adj_data, data_adj_agg)
    return data


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

    cat_agg_rules = {c: m_rule for c in catvar_names} | {
        f"{c}_MEAN": m_rule for c in adj_catvar_names
    }
    data_agg = data.groupby("SK_ID_CURR").agg(num_agg_rules | cat_agg_rules)
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
    free(closed, closed_agg, data)

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

    data_agg = data.groupby("SK_ID_CURR").agg(num_agg_rules | cat_agg_rules)
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


def group_PCB_by_curr_v2(data, catvar_names):
    mm_rule = ["max", "mean"]
    mms_rule = mm_rule + ["size"]
    agg_rules = {
        "MONTHS_BALANCE": mms_rule,
        "SK_DPD": mm_rule,
        "SK_DPD_DEF": mm_rule,
    } | {c: ["mean"] for c in catvar_names}
    grouped = data.groupby("SK_ID_CURR")
    aggregated = grouped.agg(agg_rules)
    flatten_and_prefix_columns(aggregated, "POS")

    # Count pos cash accounts
    aggregated["POS_COUNT"] = grouped.size()

    # Free data
    free(data)

    return aggregated


def group_CCB_by_curr_v2(data, catvar_names):
    ## TODO : Pas à sa place, c'est du nettoyage : regrouper les clean
    data.drop(['SK_ID_PREV'], axis=1, inplace=True)
    
    agg_rule = ["min", "max", "mean", "sum", "var"]
    grouped = data.groupby('SK_ID_CURR')
    aggregated = grouped.agg(agg_rule)
    flatten_and_prefix_columns(aggregated, "CC")

    # Count installments accounts
    aggregated["CC_COUNT"] = grouped.size()
    
    # Free data
    free(data)

    return aggregated


def group_IP_by_curr_v2(data, catvar_names):
    # Features: Perform aggregations
    nu = ["nunique"]
    mms = ["max", "mean", "sum"]
    mmsv = mms + ["var"]
    mmms = ["min"] + mms
    agg_rules = {
        'NUM_INSTALMENT_VERSION': nu,
        'DPD': mms,
        'DBD': mms,
        'PAYMENT_PERC': mmsv,
        'PAYMENT_DIFF': mmsv,
        'AMT_INSTALMENT': mms,
        'AMT_PAYMENT': mmms,
        'DAYS_ENTRY_PAYMENT': mms,
    } | {cat: ["mean"] for cat in catvar_names}
    grouped = data.groupby('SK_ID_CURR')
    aggregated = grouped.agg(agg_rules)
    flatten_and_prefix_columns(aggregated, "INSTAL")

    aggregated["INSTAL_COUNT"] = grouped.size()
    return aggregated


""" Feature engineering
"""


def add_A_derived_features_v2(data):
    """DEPRECATED use `add_derived_features` instead"""
    add_derived_features(data,
        """
        DAYS_EMPLOYED_PERC = DAYS_EMPLOYED / DAYS_BIRTH
        INCOME_CREDIT_PERC = AMT_INCOME_TOTAL / AMT_CREDIT
        INCOME_PER_PERSON = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS
        ANNUITY_INCOME_PERC = AMT_ANNUITY / AMT_INCOME_TOTAL
        PAYMENT_RATE = AMT_ANNUITY / AMT_CREDIT
        """
    )


def add_B_derived_features_v2(data):
    """DEPRECATED use `add_derived_features` instead"""
    pass


def add_PCB_derived_features_v2(data):
    """DEPRECATED use `add_derived_features` instead"""
    pass


def add_CCB_derived_features_v2(data):
    """DEPRECATED use `add_derived_features` instead"""
    pass


def add_PA_derived_features_v2(data):
    """DEPRECATED use `add_derived_features` instead"""
    add_derived_features(
        data,
        # Add feature: value ask / value received percentage
        """APP_CREDIT_PERC = AMT_APPLICATION / AMT_CREDIT"""
    )


def add_IP_derived_features_v2(data):
    """DEPRECATED use `add_derived_features` instead"""
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


""" Complete preprocessing
"""


def application_train_test_v2(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    """DEPRECATED use `preprocessed_application` instead"""
    data = load_A_table_v2(nrows, silly_mode=True)
    clean_A_cats_v2(data)
    encode_A_bin_cats_v2(data)
    data, _ = hot_encode_cats(
        data,
        dummy_na=nan_as_category,
        discard_constants=False
    )
    clean_A_nums_v2(data)
    add_A_derived_features_v2(data)
    return data


def bureau_and_balance_v2(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    """DEPRECATED use `preprocessed_bureau_and_balance` instead"""
    data, adj_data = load_B_tables_v2(nrows)
    data, cat_vars = hot_encode_cats(data, nan_as_category)
    adj_data, adj_cat_vars = hot_encode_cats(adj_data, nan_as_category)
    data = group_BB_by_bur_and_join_to_B_v2(data, adj_data, adj_cat_vars)
    return group_B_by_curr_v2(data, cat_vars, adj_cat_vars)


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
    add_PA_derived_features_v2(data)
    return group_PA_by_curr_v2(data, cats)


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
    add_IP_derived_features_v2(data)
    return group_IP_by_curr_v2(data, cats)
