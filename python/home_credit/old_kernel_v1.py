import gc

import pandas as pd
import numpy as np


def free(*objs: object) -> None:
    """DEPRECATED Not pythonic
    Free memory used by objects and garbage collect.

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


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


""" Loading
"""


def load_A_table_v1(nrows=None):
    """DEPRECATED use `load_A_table_v2` instead.
    Load `application` table."""
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


def load_B_tables_v1(nrows=None):
    """DEPRECATED use `load_B_table_v2` instead.
    Load `bureau` and `bureau_balance` tables."""
    # Note that selecting the first `nrows` rows respectively from the two tables 
    # does not guarantee that their `SK_ID_BUREAU` will have a non-empty intersection,
    # so joining them with `merge` or `join` may result in significant loss.
    data = pd.read_csv('../../dataset/csv/bureau.csv', nrows=nrows)
    adj_data = pd.read_csv('../../dataset/csv/bureau_balance.csv', nrows=nrows)
    return data, adj_data


def load_PA_table_v1(nrows=None):
    """DEPRECATED use `load_PA_table_v2` instead.
    Load `previous_application` table."""
    return pd.read_csv('../../dataset/csv/previous_application.csv', nrows=nrows)


def load_PCB_table_v1(nrows=None):
    """DEPRECATED use `load_PCB_table_v2` instead.
    Load `pos_cash_balance` table."""
    return pd.read_csv('../../dataset/csv/POS_CASH_balance.csv', nrows=nrows)


def load_CCB_table_v1(nrows=None):
    """DEPRECATED use `load_CCB_table_v2` instead.
    Load `credit_card_balance` table."""
    return pd.read_csv('../../dataset/csv/credit_card_balance.csv', nrows=nrows)


def load_IP_table_v1(nrows=None):
    """DEPRECATED use `load_IP_table_v2` instead.
    Load `installments_payments` table."""
    return pd.read_csv('../../dataset/csv/installments_payments.csv', nrows=nrows)


""" Cleaning
"""


def clean_A_cats_v1(data):
    """DEPRECATED use `clean_A_cats_v2` instead"""
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    # NB > copy() was added to avoid subsequent errors or warnings caused by working on a view
    return data[data['CODE_GENDER'] != 'XNA'].copy()


def clean_A_nums_v1(data):
    """DEPRECATED use `clean_A_nums_v2` instead"""
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)


def clean_PA_nums_v1(data):
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    data['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    data['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    data['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    data['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    data['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)


def clean_A_data_v1(data):
    pass


def clean_B_data_v1(data):
    pass


def clean_PA_data_v1(data):
    pass


def clean_PCB_data_v1(data):
    pass


def clean_CCB_data_v1(data):
    pass


def clean_IP_data_v1(data):
    pass


""" Encoding
"""


def encode_A_bin_cats_v1(data):    
    """DEPRECATED use `clean_IP_cats_v2` instead"""
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        # Unused `uniques` has been replaced by `_`
        data[bin_feature], _ = pd.factorize(data[bin_feature])


""" Grouping
"""


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
    cat_aggregations = {cat: ['mean'] for cat in catvar_names}
    for cat in adj_catvar_names:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    data_agg = data.groupby('SK_ID_CURR').agg(num_aggregations | cat_aggregations)
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
    cat_aggregations = {cat: ['mean'] for cat in catvar_names}
    data_agg = data.groupby('SK_ID_CURR').agg(num_aggregations | cat_aggregations)
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


""" Feature engineering
"""


def add_A_derived_features_v1(data):
    """DEPRECATED use `add_A_derived_features` instead"""
    # Some simple new features (percentages)
    data['DAYS_EMPLOYED_PERC'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
    data['INCOME_CREDIT_PERC'] = data['AMT_INCOME_TOTAL'] / data['AMT_CREDIT']
    data['INCOME_PER_PERSON'] = data['AMT_INCOME_TOTAL'] / data['CNT_FAM_MEMBERS']
    data['ANNUITY_INCOME_PERC'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
    data['PAYMENT_RATE'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']


def add_B_derived_features_v1(data):
    pass


def add_PCB_derived_features_v1(data):
    # TODO Après agrégation suivant SK_ID_CURR
    # Count pos cash accounts
    # pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    pass


def add_CCB_derived_features_v1(data):
    pass


def add_PA_derived_features_v1(data):
    """DEPRECATED use `add_PA_derived_features` instead"""
    # Add feature: value ask / value received percentage
    data['APP_CREDIT_PERC'] = data['AMT_APPLICATION'] / data['AMT_CREDIT']


def add_IP_derived_features_v1(data):
    # TODO Après agrégation suivant SK_ID_CURR
    # Count installments accounts
    # ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    """DEPRECATED use `add_IP_derived_features` instead"""
    # Percentage and difference paid in each installment (amount paid and installment value)
    data['PAYMENT_PERC'] = data['AMT_PAYMENT'] / data['AMT_INSTALMENT']
    data['PAYMENT_DIFF'] = data['AMT_INSTALMENT'] - data['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    data['DPD'] = data['DAYS_ENTRY_PAYMENT'] - data['DAYS_INSTALMENT']
    data['DBD'] = data['DAYS_INSTALMENT'] - data['DAYS_ENTRY_PAYMENT']
    data['DPD'] = data['DPD'].apply(lambda x: max(x, 0))
    data['DBD'] = data['DBD'].apply(lambda x: max(x, 0))


""" Complete preprocessing
"""


def application_train_test_v1(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    """DEPRECATED use `application_train_test_v2` instead"""
    data = load_A_table_v1(nrows)
    data = clean_A_cats_v1(data)
    encode_A_bin_cats_v1(data)
    data, _ = one_hot_encoder(data, nan_as_category)
    clean_A_nums_v1(data)
    add_A_derived_features_v1(data)
    return data


def bureau_and_balance_v1(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    """DEPRECATED use `bureau_and_balance_v2` instead"""
    data, adj_data = load_B_tables_v1(nrows)
    data, catvars = one_hot_encoder(data, nan_as_category)
    adj_data, adj_catvars = one_hot_encoder(adj_data, nan_as_category)
    data = group_BB_by_bur_and_join_to_B_v1(data, adj_data, adj_catvars)
    return group_B_by_curr_v1(data, catvars, adj_catvars)


def previous_application_v1(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    data = load_PA_table_v1(nrows)
    data, cats = one_hot_encoder(data, nan_as_category)
    clean_PA_nums_v1(data)
    add_PA_derived_features_v1(data)
    return group_PA_by_curr_v1(data, cats)


def pos_cash_balance_v1(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    data = load_PCB_table_v1(nrows)
    data, cats = one_hot_encoder(data, nan_as_category)
    return group_PCB_by_curr_v1(data, cats)


def credit_card_balance_v1(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    data = load_CCB_table_v1(nrows)
    data, cats = one_hot_encoder(data, nan_as_category)
    return group_CCB_by_curr_v1(data, cats)


def installments_payments_v1(
    nrows=None,
    nan_as_category=False
) -> pd.DataFrame:
    data = load_IP_table_v1(nrows)
    data, cats = one_hot_encoder(data, nan_as_category)
    add_IP_derived_features_v1(data)
    return group_IP_by_curr_v1(data, cats)

