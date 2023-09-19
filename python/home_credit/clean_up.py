from pepper.utils import get_weekdays
from pepper.db_utils import cast_columns
from pepper.feat_eng import nullify
from pepper.cache import Cache

from home_credit.load import load_raw_table, get_table
from home_credit.merge import currentize, targetize
from home_credit.feat_eng import negate_numerical_data
from home_credit.impute import impute_credit_card_balance_drawings

from home_credit.cols_map import get_group
# from home_credit.tables import Application


import pandas as pd
import numpy as np


def targetize_table(data: pd.DataFrame) -> None:
    # sourcery skip: pandas-avoid-inplace
    """
    DEPRECATED Use home_credit.tables.targetize_table instead
    
    Targetize and currentize the DataFrame if necessary.

    This function checks if the DataFrame requires targetization and currentization
    and applies these transformations if needed.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to perform targetization and currentization on.

    Returns
    -------
    None

    Example
    -------
    >>> _targetize(data)
    
    Notes
    -----
    - The 'application' table is inherently targetized and currentized.
    - The 'bureau_balance' table is the only one not currentized.
    """
    # Check if the table requires currentization, e.g., 'bureau_balance'
    if "SK_ID_CURR" not in data:
        currentize(data)
        data.dropna(inplace=True)
        cast_columns(data, "SK_ID_CURR", np.uint32)

    if "TARGET" not in data:
        targetize(data)


def get_clean_bureau_balance() -> pd.DataFrame:
    def _clean_up_bureau_balance_status(data):
        col = "STATUS"
        data[col] = data[col].replace("X", np.nan)
        data[col] = data[col].fillna(method="ffill")
        # data[col] = data[col].replace("C", 0)
        # cast_columns(data, "STATUS", np.uint8)

    def _get_clean_bureau_balance() -> pd.DataFrame:
        data = load_raw_table("bureau_balance")
        targetize_table(data)
        cast_columns(data, ["SK_ID_BUREAU", "SK_ID_CURR"], np.uint32)
        data.MONTHS_BALANCE = -data.MONTHS_BALANCE
        cast_columns(data, "MONTHS_BALANCE", np.uint8)
        data = data.sort_values(by=["SK_ID_BUREAU", "MONTHS_BALANCE"])
        _clean_up_bureau_balance_status(data)
        data.set_index(["SK_ID_BUREAU", "MONTHS_BALANCE"], inplace=True)
        data.columns.name = "CLEAN_BUREAU_BALANCE"
        return data
    
    return Cache.init(
        "clean_bureau_balance",
        _get_clean_bureau_balance
    )


def get_clean_bureau_balance_with_na_current() -> pd.DataFrame:
    def _get_clean_bureau_balance_with_na_current() -> pd.DataFrame:
        data = load_raw_table("bureau_balance")
        targetize_table(data)
        data = data[data.SK_ID_CURR.isna()].copy()
        data.drop(columns="SK_ID_CURR", inplace=True)
        cast_columns(data, "SK_ID_BUREAU", np.uint32)
        data.MONTHS_BALANCE = -data.MONTHS_BALANCE
        cast_columns(data, "MONTHS_BALANCE", np.uint8)
        data = data.sort_values(by=["SK_ID_BUREAU", "MONTHS_BALANCE"])
        return data
    
    return Cache.init(
        "clean_bureau_balance_with_na_current",
        _get_clean_bureau_balance_with_na_current
    )


def get_clean_installments_payments() -> pd.DataFrame:
    def _get_clean_installments_payments() -> pd.DataFrame:
        data = load_raw_table("installments_payments")
        targetize_table(data)
        cast_columns(data, ["SK_ID_PREV", "SK_ID_CURR"], np.uint32)
        cast_columns(data, ["NUM_INSTALMENT_VERSION"], np.uint8)
        negate_numerical_data(data.DAYS_INSTALMENT)
        negate_numerical_data(data.DAYS_ENTRY_PAYMENT)
        cast_columns(data, ["NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"], np.uint16)
        cast_columns(data, ["AMT_INSTALMENT", "AMT_PAYMENT"], np.float16)
        return data[list(data.columns[:6]) + ["AMT_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_PAYMENT"]]
    
    return Cache.init(
        "clean_installments_payments",
        _get_clean_installments_payments
    )


def get_clean_installments_payments_without_entry() -> pd.DataFrame:
    def _get_clean_installments_payments_without_entry() -> pd.DataFrame:
        data = load_raw_table("installments_payments")
        data = data[data.DAYS_ENTRY_PAYMENT.isna()].copy()
        data.drop(columns=["DAYS_ENTRY_PAYMENT", "AMT_PAYMENT"], inplace=True)
        targetize_table(data)
        cast_columns(data, ["SK_ID_PREV", "SK_ID_CURR"], np.uint32)
        cast_columns(data, ["NUM_INSTALMENT_VERSION"], np.uint8)
        negate_numerical_data(data.DAYS_INSTALMENT)
        cast_columns(data, ["NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT"], np.uint16)
        cast_columns(data, ["AMT_INSTALMENT"], np.float16)
        return data
    
    return Cache.init(
        "clean_installments_payments_without_entry",
        _get_clean_installments_payments_without_entry
    )


def get_clean_pos_cash_balance() -> pd.DataFrame:
    def _get_clean_pos_cash_balance() -> pd.DataFrame:
        data = load_raw_table("pos_cash_balance")
        targetize_table(data)
        nullify(data.NAME_CONTRACT_STATUS, "XNA")
        negate_numerical_data(data.MONTHS_BALANCE)
        data.dropna(inplace=True)
        cast_columns(data, ["SK_ID_PREV", "SK_ID_CURR"], np.uint32)
        cast_columns(data, ["MONTHS_BALANCE"], np.uint8)
        cast_columns(data, ["SK_DPD", "SK_DPD_DEF"], np.uint16)
        cast_columns(data, ["CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE"], np.uint8)
        return data
    
    return Cache.init(
        "clean_pos_cash_balance",
        _get_clean_pos_cash_balance
    )


def get_clean_pos_cash_balance_with_na() -> pd.DataFrame:
    def _get_clean_pos_cash_balance_with_na() -> pd.DataFrame:
        data = load_raw_table("pos_cash_balance")
        targetize_table(data)
        nullify(data.NAME_CONTRACT_STATUS, "XNA")
        negate_numerical_data(data.MONTHS_BALANCE)
        data = data[data.isna().any(axis=1)].copy()
        cast_columns(data, ["SK_ID_PREV", "SK_ID_CURR"], np.uint32)
        cast_columns(data, ["MONTHS_BALANCE"], np.uint8)
        cast_columns(data, ["SK_DPD", "SK_DPD_DEF"], np.uint16)
        data.CNT_INSTALMENT.fillna(-1, inplace=True)
        data.CNT_INSTALMENT_FUTURE.fillna(-1, inplace=True)
        cast_columns(data, ["CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE"], np.int8)
        return data
    
    return Cache.init(
        "clean_pos_cash_balance_with_na",
        _get_clean_pos_cash_balance_with_na
    )


def get_clean_credit_card_balance() -> pd.DataFrame:
    def _get_clean_credit_card_balance() -> pd.DataFrame:
        data = load_raw_table("credit_card_balance")
        targetize_table(data)
        negate_numerical_data(data.MONTHS_BALANCE)
        impute_credit_card_balance_drawings(data)
        return data
    
    return Cache.init(
        "clean_credit_card_balance",
        _get_clean_credit_card_balance
    )


def get_clean_bureau() -> pd.DataFrame:
    
    def _load():
        return load_raw_table("bureau")
    
    def _impute(data: pd.DataFrame) -> None:
        pass
    
    def _encode(data: pd.DataFrame):
        ages_cols = get_group("bureau", "ages")[:-1]
        data[ages_cols] = -data[ages_cols]
    
    def _downcast(data: pd.DataFrame):
        cast_columns(data, ["SK_ID_CURR", "SK_ID_BUREAU"], np.uint32)
        cast_columns(data, "CNT_CREDIT_PROLONG", np.uint8)
        cast_columns(data, "DAYS_CREDIT_UPDATE", np.int32)
        # cast_columns(data, "DAYS_CREDIT_ENDDATE", np.int32) NAs
        cast_columns(data, "DAYS_CREDIT", np.uint16)
        # cast_columns(data, "DAYS_ENDDATE_FACT", np.uint16) NAs
        cast_columns(data, "CREDIT_DAY_OVERDUE", np.uint16)
        # Downcast float32 du groupe financial_statement impossible, tous des NA sauf 1

    def _get_clean_table() -> pd.DataFrame:
        # Load the table
        data = _load()

        # Targetize the table, which may involve currentization
        targetize_table(data)

        # Impute missing values and correct outliers
        _impute(data)

        # Map categories
        _encode(data)

        # Downcast data types, especially float64 with no missing values
        _downcast(data)
        
        data.set_index(["SK_ID_CURR", "SK_ID_BUREAU"], inplace=True)
        data.columns.name = "CLEAN_BUREAU"

        return data

    return Cache.init("clean_bureau", _get_clean_table)


def get_clean_previous_application() -> pd.DataFrame:
    def _load():
        return load_raw_table("previous_application")

    def _impute(data: pd.DataFrame) -> None:
        cols = get_group("previous_application", "ages")
        data[cols] = data[cols].replace(365_243, 0)
        data[cols] = -data[cols]
    
    def _encode(data: pd.DataFrame):
        # FLAGS : ('Y', 'N') -> (0, 1)
        cols = "FLAG_LAST_APPL_PER_CONTRACT"
        to_replace = {"Y": 1, "N": 0}
        data[cols] = data[cols].replace(to_replace)

    def _downcast(data: pd.DataFrame):
        cast_columns(data, ["SK_ID_PREV", "SK_ID_CURR"], np.uint16)
        flags_cols = ["FLAG_LAST_APPL_PER_CONTRACT", "NFLAG_LAST_APPL_IN_DAY"]
        cast_columns(data, flags_cols, np.uint8)
        cast_columns(data, "SELLERPLACE_AREA", np.uint16)

    def _get_clean_table() -> pd.DataFrame:
        # Load the table
        data = _load()

        # Targetize the table, which may involve currentization
        targetize_table(data)

        # Impute missing values and correct outliers
        _impute(data)

        # Map categories
        _encode(data)

        # Downcast data types, especially float64 with no missing values
        _downcast(data)

        data.set_index(["SK_ID_CURR", "SK_ID_PREV"], inplace=True)
        data.columns.name = "CLEAN_PREVIOUS_APPLICATION"
        
        return data

    return Cache.init("clean_previous_application", _get_clean_table)


def get_clean_application() -> pd.DataFrame:
    
    def _load():
        # On utilise `get_table` qui implique le cache
        # TODO Pour utiliser `get_raw`, il faudrait sauvegarder
        # la version juxtaposée des tables `application_test` et `application_train`
        return get_table("application")

    # TODO : pour les fonctions suivantes,
    # tendre vers une description compète en JSON des règles de transformation
    def _impute(data: pd.DataFrame) -> None:
        """
        Impute missing values and perform data corrections.

        This function imputes missing values and performs data corrections in the
        provided DataFrame based on predefined rules.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame to perform imputation and corrections on.

        Returns
        -------
        None

        Example
        -------
        >>> _impute(data)
        """
        # Round an outlier value (408583) to the nearest integer
        cols = "DAYS_REGISTRATION"
        data[cols] = data[cols].round()

        # Fill a specific missing value case (118330) with 0
        cols = "DAYS_LAST_PHONE_CHANGE"
        data[cols] = data[cols].fillna(0)

        # Fill missing values (148605 and 317181) with 1
        cols = "CNT_FAM_MEMBERS"
        data[cols] = data[cols].fillna(1)

        # Correct the employment days for retirees to be 0 instead of 365243
        cols = "DAYS_EMPLOYED"
        data[cols] = data[cols].replace(365_243, 0)

        # Correct a specific case (224393) of REGION_RATING_CLIENT_W_CITY to 2
        # instead of -1
        cols = "REGION_RATING_CLIENT_W_CITY"
        data[cols] = data[cols].replace(-1, 2)

        # Fill missing values in SOCIAL_CIRCLE with zeros
        # cols = Application.cols_group("social_circle_counts")
        cols = get_group("application", "social_circle_counts")
        data[cols] = data[cols].fillna(0)

        # Code missing values in AMT_REQ_CREDIT_BUREAU as -1 for preprocessing,
        # to avoid float64, but they should be reprocessed before model training
        # cols = Application.cols_group("credit_bureau_request_counts")
        cols = get_group("application", "credit_bureau_request_counts")
        data[cols] = data[cols].fillna(-1)


    def _encode(data: pd.DataFrame):
        """
        Encode categorical variables into numeric representations.

        This function encodes categorical variables in the DataFrame into numeric
        representations based on predefined encoding rules.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame to perform encoding on.

        Returns
        -------
        None

        Example
        -------
        >>> _encode(data)
        """
        # Reverse the sign of columns in the "ages" group
        # cols = Application.cols_group("ages")
        cols = get_group("application", "ages")
        data[cols] = -data[cols]

        # Encode the weekday as an integer starting from MONDAY=0
        cols = "WEEKDAY_APPR_PROCESS_START"
        to_replace = {d: i for i, d in enumerate(get_weekdays())}
        data[cols] = data[cols].replace(to_replace)

        # Convert gender to numeric values: M = 0, F = 1, XNA = 2
        cols = "CODE_GENDER"
        to_replace = {"M": 0, "F": 1, "XNA": 2}
        data[cols] = data[cols].replace(to_replace)

        # Convert ownership flags to numeric values: Y = 1, N = 0
        #cols = Application.cols_group("ownership_flags")
        cols = get_group("application", "ownership_flags")
        to_replace = {"Y": 1, "N": 0}
        data[cols] = data[cols].replace(to_replace)


    def _downcast(data: pd.DataFrame):
        """
        Downcast numeric columns in the DataFrame.

        This function downcasts numeric columns to more memory-efficient data types,
        such as uint8, uint16, int8, int16, etc., where applicable. It helps reduce
        memory usage without losing data precision.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame to perform downcasting on.

        Returns
        -------
        None

        Example
        -------
        >>> _downcast(data)
        """
        # Cast columns to more memory-efficient data types based on groups
        # TODO : en attendant l'intégration en full OO, je court circuite
        # l'invocation de la classe dans cette méthode
        # def cast_cols_group(data: pd.DataFrame, group_name: str, dtype: np.dtype):
        #    cast_columns(data, Application.cols_group(group_name), dtype)
        def cast_cols_group(data: pd.DataFrame, group_name: str, dtype: np.dtype):
            cols = get_group("application", group_name)
            cast_columns(data, cols, dtype)

        # Apply downcasting for various column groups
        cast_cols_group(data, "target", np.uint8)
        cast_cols_group(data, "keys", np.uint32)
        cast_cols_group(data, "gender", np.uint8)
        cast_cols_group(data, "financial_statement", np.float32)
        cast_cols_group(data, "contact_flags", np.uint8)
        cast_cols_group(data, "commute_flags", np.uint8)
        cast_cols_group(data, "ownership_flags", np.uint8)

        # Downcast flag document columns
        flag_doc_cols = list(data.columns[data.columns.str.startswith("FLAG_DOCUMENT")])
        cast_columns(data, flag_doc_cols, np.uint8)

        cast_cols_group(data, "family_counts", np.uint8)
        cast_cols_group(data, "process_start", np.uint8)
        cast_cols_group(data, "region_ratings", np.uint8)

        # Downcast age-related columns
        cast_cols_group(data, "ages", np.uint16)

        # Downcast credit bureau request count columns
        # cnt_req_cols = Application.cols_group("credit_bureau_request_counts")
        cnt_req_cols = get_group("application", "credit_bureau_request_counts")
        cnt_req_cols.remove("AMT_REQ_CREDIT_BUREAU_QRT")
        cast_columns(data, cnt_req_cols, np.int8)
        cast_columns(data, "AMT_REQ_CREDIT_BUREAU_QRT", np.int16)


    def _get_clean_table() -> pd.DataFrame:
        """
        Get a cleaned and prepared table.

        This function loads the raw data table, performs target encoding (if applicable),
        handles missing values, corrects outliers, encodes categorical features, and
        downcasts data types. The resulting table is cleaned and ready for analysis.

        Returns
        -------
        pd.DataFrame
            A cleaned and prepared DataFrame.

        Example
        -------
        >>> clean_data = _get_clean_table()
        """
        # Load the table
        data = _load()

        # Targetize the table, which may involve currentization
        targetize_table(data)

        # Impute missing values and correct outliers
        _impute(data)

        # Map categories
        _encode(data)

        # Downcast data types, especially float64 with no missing values
        _downcast(data)

        data.set_index("SK_ID_CURR", inplace=True)
        data.columns.name = "CLEAN_APPLICATION"
        
        return data

    return Cache.init("clean_application", _get_clean_table)