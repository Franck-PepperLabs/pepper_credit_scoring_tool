from typing import List, Dict, Tuple, Union, Optional

from pepper.db_utils import cast_columns
from pepper.utils import get_weekdays

from home_credit.load import load_raw_table, _load_application
from home_credit.persist import this_f_name, controlled_load
from home_credit.cols_map import get_cols_map

from home_credit.impute import impute_credit_card_balance_drawings

# from home_credit.clean_up import *

import pandas as pd
import numpy as np


class HomeCreditTable:
    """
    Base class for tables in the Home Credit dataset.

    Subclasses should define the 'name' attribute specifying the table name.
    """

    name = None

    @classmethod
    def raw(cls) -> pd.DataFrame:
        """
        Load the raw data for the table.

        Returns
        -------
        pd.DataFrame
            The raw data for the table as a DataFrame.

        Raises
        ------
        ValueError
            If the 'name' attribute is not defined in the subclass.
        """
        if cls.name is None:
            raise ValueError(f"Subclass {cls.__name__} must define 'name' attribute.")
        return load_raw_table(cls.name)

    @classmethod
    def cols_map(cls, group_name: str) -> Dict[str, Tuple[str]]:
        """
        Get the column mappings for a specified group.

        Parameters
        ----------
        group_name : str
            The name of the group for which to retrieve column mappings.

        Returns
        -------
        Dict[str, Tuple[str]]
            A dict containing the column mappings for the specified group.

        Raises
        ------
        ValueError
            If the 'name' attribute is not defined in the subclass.

        Notes
        -----
        The column mappings are specified in the 'cols_map.json' file.

        Example
        -------
        >>> CreditCardBalance.cols_map("balancing")
        {
            'AMT_BALANCE': ('BAL', '', 'AMT'),
            'AMT_DRAWINGS_CURRENT': ('DRW', 'TOT', 'AMT'),
            'AMT_RECEIVABLE_PRINCIPAL': ('RCV', '', 'AMT'),
            'AMT_TOTAL_RECEIVABLE': ('RCV', 'TOT', 'AMT'),
            'AMT_PAYMENT_CURRENT': ('PYT', '', 'AMT'),
            'AMT_PAYMENT_TOTAL_CURRENT': ('PYT', 'TOT', 'AMT')
        }
        """
        if cls.name is None:
            raise ValueError(f"Subclass {cls.__name__} must define 'name' attribute.")
        return get_cols_map(cls.name, group_name)

    @classmethod
    def cols_group(cls,
        group_name: str,
        shorten: bool = False
    ) -> Union[List[str], List[Tuple[str]]]:
        """
        Get the column names and subgroups for a specified group.

        Parameters
        ----------
        group_name : str
            The name of the group for which to retrieve column names and subgroups.
        shorten : bool, optional (default=False)
            If True, return only the values (column names) of the column mappings.

        Returns
        -------
        Union[List[str], List[Tuple[str]]]
            A list containing the column names and subgroups for the specified group.

        Notes
        -----
        The column mappings are specified in the 'cols_map.json' file.

        Example
        -------
        >>> CreditCardBalance.cols_group("balancing")
        [
            'AMT_BALANCE',
            'AMT_DRAWINGS_CURRENT',
            'AMT_RECEIVABLE_PRINCIPAL',
            'AMT_TOTAL_RECEIVABLE',
            'AMT_PAYMENT_CURRENT',
            'AMT_PAYMENT_TOTAL_CURRENT'
        ]
        >>> CreditCardBalance.cols_group("balancing", shorten=True)
        [
            ('BAL', '', 'AMT'),
            ('DRW', 'TOT', 'AMT'),
            ('RCV', '', 'AMT'),
            ('RCV', 'TOT', 'AMT'),
            ('PYT', '', 'AMT'),
            ('PYT', 'TOT', 'AMT')
        ]
        """
        cmap = cls.cols_map(group_name)
        return list(cmap.values() if shorten else cmap.keys())

    @classmethod
    def cast_cols_group(cls,
        data: pd.DataFrame,
        group_name: str,
        dtype: np.dtype
    ) -> None:
        """Cast columns to more memory-efficient data types based on groups"""
        cast_columns(data, cls.cols_group(group_name), dtype)


    """ Cleaning
    """

    @classmethod
    def _impute(cls, data: pd.DataFrame) -> None:
        """Impute missing values and correct outliers"""
        pass

    @classmethod
    def _encode(cls, data: pd.DataFrame) -> None:
        """Map categories"""
        pass

    @classmethod
    def _downcast(cls, data: pd.DataFrame) -> None:
        """Downcast data types, especially float64 with no missing values"""
        pass

    @classmethod
    def _reset_index(cls, data: pd.DataFrame) -> None:
        """Sort and reset the index and rename columns index"""
        keys = cls.cols_group("keys")
        data = data.sort_values(by=keys)
        data = data.set_index(keys)
        data.columns.name = f"CLEAN_{cls.name.upper()}"

    @classmethod
    def _get_clean_table(cls) -> pd.DataFrame:
        # Load the raw table
        data = cls.raw()

        # Cleaning pipeline
        targetize_table(data)  # Targetize (and currentize) the table if necessary
        cls._impute(data)  # Impute missing values and correct outliers
        cls._encode(data)  # Map categories
        cls._downcast(data)  # Downcast data types
        cls._reset_index(data)  # Sort end reset the index and rename columns index

        return data

    @classmethod
    def clean(cls,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.DataFrame:
        return controlled_load(
            this_f_name(), locals().copy(),
            cls._get_clean_table, f"clean_{cls.name}",
            True
        )


class BureauBalance(HomeCreditTable):

    name = "bureau_balance"

    """ Options NA included or not
    @staticmethod
    def clean_with_na_current() -> pd.DataFrame:
        return get_clean_bureau_balance_with_na_current()
    """

    @classmethod
    def _encode(cls, data: pd.DataFrame) -> None:
        # TODO : abstraction : config dans un fichier
        data.MONTHS_BALANCE = -data.MONTHS_BALANCE
        data = data.sort_values(by=["SK_ID_BUREAU", "MONTHS_BALANCE"])
        data.STATUS = data.STATUS.replace("X", np.nan)
        data.STATUS = data.STATUS.fillna(method="ffill")
        # data.STATUS = data.STATUS.replace("C", 0)

    @classmethod
    def _downcast(cls, data: pd.DataFrame) -> None:
        # TODO : abstraction : config dans un fichier

        # Apply downcasting for various column groups
        cast_columns(data, ["SK_ID_BUREAU", "SK_ID_CURR"], np.uint32)
        cast_columns(data, ["TARGET", "MONTHS_BALANCE"], np.uint8)
        # cast_columns(data, "STATUS", np.uint8)


class InstallmentsPayments(HomeCreditTable):

    name = "installments_payments"

    """ Options No entry included or not
    @staticmethod
    def clean_without_entry() -> pd.DataFrame:
        return get_clean_installments_payments_without_entry()
    """

    @classmethod
    def _encode(cls, data: pd.DataFrame) -> None:
        # TODO : abstraction : config dans un fichier
        ages_cols = InstallmentsPayments.cols_group("ages")
        data[ages_cols] = -data[ages_cols]

    @classmethod
    def _downcast(cls, data: pd.DataFrame) -> None:
        # TODO : abstraction : config dans un fichier

        # Apply downcasting for various column groups
        cast_columns(data, ["SK_ID_PREV", "SK_ID_CURR"], np.uint32)
        cast_columns(data, ["TARGET", "NUM_INSTALMENT_VERSION"], np.uint8)
        cast_columns(data, [
            "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"
        ], np.uint16)
        cast_columns(data, ["AMT_INSTALMENT", "AMT_PAYMENT"], np.float16)


class POSCashBalance(HomeCreditTable):

    name = "pos_cash_balance"

    """ Options NA included, no_NA included
    @staticmethod
    def clean_with_na() -> pd.DataFrame:
        return get_clean_pos_cash_balance_with_na()
    """

    @classmethod
    def _impute(cls, data: pd.DataFrame) -> None:
        """Impute missing values and correct outliers"""
        pass

    @classmethod
    def _encode(cls, data: pd.DataFrame) -> None:
        # TODO : abstraction : config dans un fichier
        data.MONTHS_BALANCE = -data.MONTHS_BALANCE

    @classmethod
    def _downcast(cls, data: pd.DataFrame) -> None:
        # TODO : abstraction : config dans un fichier

        # Apply downcasting for various column groups
        cast_columns(data, ["SK_ID_PREV", "SK_ID_CURR"], np.uint32)
        cast_columns(data, ["MONTHS_BALANCE"], np.uint8)
        cast_columns(data, ["SK_DPD", "SK_DPD_DEF"], np.uint16)
        cast_columns(data, ["CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE"], np.int8)


class CreditCardBalance(HomeCreditTable):

    name = "credit_card_balance"

    @classmethod
    def _impute(cls, data: pd.DataFrame) -> None:
        """Impute missing values and correct outliers"""
        impute_credit_card_balance_drawings(data)

    @classmethod
    def _encode(cls, data: pd.DataFrame) -> None:
        # TODO : abstraction : config dans un fichier
        data.MONTHS_BALANCE = -data.MONTHS_BALANCE

    @classmethod
    def _downcast(cls, data: pd.DataFrame) -> None:
        # TODO : abstraction : config dans un fichier

        # Apply downcasting for various column groups
        cast_columns(data, ["SK_ID_PREV", "SK_ID_CURR"], np.uint32)
        cast_columns(data, ["TARGET", "MONTHS_BALANCE"], np.uint8)


class Bureau(HomeCreditTable):

    name = "bureau"

    @staticmethod
    def _to_curr_map():
        curr_map = Bureau.raw()[["SK_ID_BUREAU", "SK_ID_CURR"]]
        curr_map = curr_map.set_index("SK_ID_BUREAU")
        curr_map.SK_ID_CURR.astype(np.uint32)
        return curr_map

    @classmethod
    def to_curr_map(cls,
        no_cache: Optional[bool] = False,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            Bureau._to_curr_map, "bureau_to_curr_map", True
        ).SK_ID_CURR

    @classmethod
    def _impute(cls, data: pd.DataFrame) -> None:
        """Impute missing values and correct outliers"""
        pass

    @classmethod
    def _encode(cls, data: pd.DataFrame) -> None:
        # TODO : abstraction : config dans un fichier
        ages_cols = Bureau.cols_group("ages")[:-1]
        data[ages_cols] = -data[ages_cols]

    @classmethod
    def _downcast(cls, data: pd.DataFrame) -> None:
        # TODO : abstraction : config dans un fichier

        # Apply downcasting for various column groups
        cast_columns(data, "TARGET", np.uint8)
        cast_columns(data, "SK_ID_BUREAU", np.uint32)
        cast_columns(data, "SK_ID_CURR", np.uint32)
        cast_columns(data, "CNT_CREDIT_PROLONG", np.uint8)
        cast_columns(data, "DAYS_CREDIT_UPDATE", np.int32)
        # cast_columns(data, "DAYS_CREDIT_ENDDATE", np.int32) NAs
        cast_columns(data, "DAYS_CREDIT", np.uint16)
        # cast_columns(data, "DAYS_ENDDATE_FACT", np.uint16) NAs
        cast_columns(data, "CREDIT_DAY_OVERDUE", np.uint16)
        # Downcast float32 of financial_statement is impossible : all NA (except one)


    """ Abstraction -> dans la classe abstraite HomeCreditTable
    @classmethod
    def _reset_index(cls, data: pd.DataFrame) -> None:
        data.set_index(["SK_ID_CURR", "SK_ID_BUREAU"], inplace=True)
        data.columns.name = "CLEAN_BUREAU"
    """

    """ Abstraction -> dans la classe abstraite HomeCreditTable
    @classmethod
    def clean(cls,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.DataFrame:
        return controlled_load(
            this_f_name(), locals().copy(),
            Bureau._get_clean_table, "clean_bureau",
            True
        )
    """


class PreviousApplication(HomeCreditTable):

    name = "previous_application"

    @classmethod
    def _impute(cls, data: pd.DataFrame) -> None:
        """Impute missing values and correct outliers"""
        cols = PreviousApplication.cols_group("ages")
        data[cols] = data[cols].replace(365_243, 0)

    @classmethod
    def _encode(cls, data: pd.DataFrame) -> None:
        # TODO : abstraction : config dans un fichier
        cols = PreviousApplication.cols_group("ages")
        data[cols] = -data[cols]
        # FLAGS : ('Y', 'N') -> (0, 1)
        cols = "FLAG_LAST_APPL_PER_CONTRACT"
        to_replace = {"Y": 1, "N": 0}
        data[cols] = data[cols].replace(to_replace)

    @classmethod
    def _downcast(cls, data: pd.DataFrame) -> None:
        # TODO : abstraction : config dans un fichier

        # Apply downcasting for various column groups
        Application.cast_cols_group(data, "target", np.uint8)
        Application.cast_cols_group(data, "keys", np.uint16)
        Application.cast_cols_group(data, "last_app_flags", np.uint8)
        cast_columns(data, "SELLERPLACE_AREA", np.uint16)


class Application(HomeCreditTable):

    name = "application"

    @classmethod
    def raw(cls) -> pd.DataFrame:
        """
        Load the raw data for the `application` table.

        Returns
        -------
        pd.DataFrame
            The raw data for the table as a DataFrame.

        Raises
        ------
        ValueError
            If the 'name' attribute is not defined in the subclass.
        """
        return _load_application()

    @staticmethod
    def _to_target_map():
        target_map = Application.raw()[["SK_ID_CURR", "TARGET"]]
        target_map = target_map.set_index("SK_ID_CURR")
        target_map.TARGET.astype(np.uint8)
        return target_map

    @classmethod
    def to_target_map(cls,
        no_cache: Optional[bool] = False,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            Application._to_target_map, "application_to_target_map", True
        ).TARGET

    @classmethod
    def _impute(cls, data: pd.DataFrame) -> None:
        """Impute missing values and correct outliers"""
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
        cols = Application.cols_group("social_circle_counts")
        data[cols] = data[cols].fillna(0)

        # Code missing values in AMT_REQ_CREDIT_BUREAU as -1 for preprocessing,
        # to avoid float64, but they should be reprocessed before model training
        cols = Application.cols_group("credit_bureau_request_counts")
        data[cols] = data[cols].fillna(-1)

    @classmethod
    def _encode(cls, data: pd.DataFrame) -> None:
        # Reverse the sign of columns in the "ages" group
        cols = Application.cols_group("ages")
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
        cols = Application.cols_group("ownership_flags")
        to_replace = {"Y": 1, "N": 0}
        data[cols] = data[cols].replace(to_replace)

    @classmethod
    def _downcast(cls, data: pd.DataFrame) -> None:
        # TODO : abstraction : config dans un fichier

        # Apply downcasting for various column groups
        Application.cast_cols_group(data, "target", np.uint8)
        Application.cast_cols_group(data, "keys", np.uint32)
        Application.cast_cols_group(data, "gender", np.uint8)
        Application.cast_cols_group(data, "financial_statement", np.float32)
        Application.cast_cols_group(data, "contact_flags", np.uint8)
        Application.cast_cols_group(data, "commute_flags", np.uint8)
        Application.cast_cols_group(data, "ownership_flags", np.uint8)

        # Downcast flag document columns
        # TODO abstract group definition in col_maps, here FLAG_DOCUMENT_{N}
        flag_doc_cols = list(
            data.columns[data.columns.str.startswith("FLAG_DOCUMENT")]
        )
        cast_columns(data, flag_doc_cols, np.uint8)

        Application.cast_cols_group(data, "family_counts", np.uint8)
        Application.cast_cols_group(data, "process_start", np.uint8)
        Application.cast_cols_group(data, "region_ratings", np.uint8)

        # Downcast age-related columns
        Application.cast_cols_group(data, "ages", np.uint16)

        # Downcast credit bureau request count columns
        cnt_req_cols = Application.cols_group("credit_bureau_request_counts")
        cnt_req_cols.remove("AMT_REQ_CREDIT_BUREAU_QRT")
        cast_columns(data, cnt_req_cols, np.int8)
        cast_columns(data, "AMT_REQ_CREDIT_BUREAU_QRT", np.int16)


""" Merge utils (replace the old home_credit.merge versions)
"""

def map_bur_to_curr(sk_id_bur: pd.Series) -> pd.Series:
    """
    Map `SK_ID_BUR` values to their corresponding `SK_ID_CURR` values
    by using a mapping table extracted from the 'bureau' table.

    Note - The 'bureau_balance' table needs to be currentized before it
    can be targetized.

    Parameters
    ----------
    sk_id_bur : pd.Series
        Series of `SK_ID_BUR` values.

    Returns
    -------
    pd.Series
        Series of corresponding `SK_ID_CURR` values.
    """
    return sk_id_bur.map(Bureau.to_curr_map())


def currentize(data: pd.DataFrame) -> None:
    """
    Currentize a DataFrame by adding the 'SK_ID_CURR' column based on the
    'SK_ID_BUREAU' values.

    This function adds a new column 'SK_ID_CURR' to the DataFrame by mapping
    'SK_ID_BUREAU' values to their corresponding 'SK_ID_CURR' values using
    a mapping table extracted from the 'bureau' table.

    Note: The 'bureau_balance' table needs to be currentized before it can
    be targetized.

    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame to currentize, which should contain the 'SK_ID_BUREAU' column.

    Raises:
    -------
    ValueError
        If the 'SK_ID_BUREAU' column is not found in the DataFrame.

    Returns:
    --------
    None
    """
    # Check if data contains the 'SK_ID_BUREAU' column, otherwise raise an exception
    if "SK_ID_BUREAU" not in data.columns:
        raise ValueError("The DataFrame does not contain the 'SK_ID_BUREAU' column.")
    
    data.insert(0, "SK_ID_CURR", map_bur_to_curr(data.SK_ID_BUREAU))


def map_curr_to_target(sk_id_curr: pd.Series) -> pd.Series:
    """
    Map `SK_ID_CURR` values to their corresponding `TARGET` values in the main
    table.

    Parameters
    ----------
    sk_id_curr : pd.Series
        Series of `SK_ID_CURR` values.

    Returns
    -------
    pd.Series
        Series of corresponding `TARGET` values.
    """
    return sk_id_curr.map(Application.to_target_map())


def targetize(data: pd.DataFrame) -> None:
    """
    Targetize a DataFrame by adding the 'TARGET' column based on the
    'SK_ID_CURR' values.

    This function adds a new column 'TARGET' to the DataFrame by mapping
    'SK_ID_CURR' values to their corresponding 'TARGET' values in the main table.

    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame to targetize, which should contain the 'SK_ID_CURR' column.

    Raises:
    -------
    ValueError
        If the 'SK_ID_CURR' column is not found in the DataFrame.

    Returns:
    --------
    None
    """
    # Check if data contains the 'SK_ID_CURR' column, otherwise raise an exception
    if "SK_ID_CURR" not in data.columns:
        raise ValueError("The DataFrame does not contain the 'SK_ID_CURR' column.")
    
    data.insert(0, "TARGET", map_curr_to_target(data.SK_ID_CURR))


def targetize_table(data: pd.DataFrame) -> None:
    # sourcery skip: pandas-avoid-inplace
    """
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
