from typing import List, Dict, Tuple, Union, Optional

from pepper.db_utils import cast_columns
from pepper.utils import get_weekdays

from home_credit.load import load_raw_table, _load_application
from home_credit.persist import this_f_name, controlled_load
from home_credit.cols_map import get_cols_map
from home_credit.kernel import one_hot_encode_all_cats

from home_credit.impute import impute_credit_card_balance_drawings
from home_credit.groupby import (
    get_bureau_loan_status,
    get_bureau_loan_status_by_month,
    get_bureau_loan_status_by_client_and_month,
    get_bureau_loan_status_by_client,
    
    get_bureau_loan_activity,
    get_bureau_loan_activity_by_month,
    get_bureau_loan_activity_by_client_and_month,
    
    get_bureau_mean_loan_activity,
    get_bureau_mean_loan_activity_by_client,
    
    get_rle_bureau_loan_tracking_period,
    get_rle_bureau_loan_tracking_period_by_client,
    get_rle_bureau_loan_feature_variation,
    get_rle_bureau_loan_feature_by_client_variation,
    
    get_rle_pos_cash_loan_tracking_period,
    get_rle_credit_card_loan_tracking_period,
    
    get_clean_installments_payments_base,
    get_clean_installments_payments_loan_amount,
    get_clean_installments_payments_loaned_amount,
    get_clean_installments_payments_last_version,
    get_clean_installments_payments_repaid_and_dpd,
    get_clean_installments_payments_expected_dpd_by_loan,
    get_clean_installments_payments_expected_dpd_by_client,
    get_installments_payments_by_installment,
    
    get_extended_clean_bureau
)
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
        # sourcery skip: pandas-avoid-inplace
        # -> in context, it's a very bad advice
        """Sort and reset the index and rename columns index"""
        keys = cls.cols_group("keys")
        data.sort_values(by=keys, inplace=True)
        data.set_index(keys, inplace=True)
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
            cls._get_clean_table, f"clean_{cls.name}"
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
        # sourcery skip: pandas-avoid-inplace
        # -> in context, it's a very bad advice
        # TODO : abstraction : config dans un fichier
        data.MONTHS_BALANCE = -data.MONTHS_BALANCE
        data.sort_values(by=["SK_ID_BUREAU", "MONTHS_BALANCE"], inplace=True)
        data.STATUS = data.STATUS.replace("X", np.nan)
        data.STATUS = data.STATUS.fillna(method="ffill")
        # data.STATUS = data.STATUS.replace("C", 0)

    @classmethod
    def _downcast(cls, data: pd.DataFrame) -> None:
        # TODO : abstraction : config dans un fichier
        # Apply downcasting for various column groups
        cast_columns(data, ["SK_ID_BUREAU", "SK_ID_CURR"], np.uint32)
        cast_columns(data, "MONTHS_BALANCE", np.uint8)
        # cast_columns(data, "STATUS", np.uint8)

    @classmethod
    def loan_status_by_month(cls,
        decimals: Optional[int] = 2,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_bureau_loan_status_by_month(
                BureauBalance.clean(), decimals
            ),
            "bureau_loan_status_by_month"
        )

    @classmethod
    def loan_status_by_client_and_month(cls,
        decimals: Optional[int] = 2,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_bureau_loan_status_by_client_and_month(
                BureauBalance.clean(),
                decimals
            ),
            "bureau_loan_status_by_client_and_month"
        )

    @classmethod
    def loan_status(cls,
        alpha: Optional[float] = 1,
        decimals: Optional[int] = 2,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_bureau_loan_status(
                BureauBalance.clean(), alpha, decimals
            ),
            "bureau_loan_status"
        )

    @classmethod
    def loan_activity(cls,
        alpha: Optional[float] = 1,
        decimals: Optional[int] = 2,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_bureau_loan_activity(
                BureauBalance.clean(), alpha, decimals
            ),
            "bureau_loan_status"
        )

    @classmethod
    def loan_status_by_client(cls,
        alpha: Optional[float] = 1,
        decimals: Optional[int] = 2,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_bureau_loan_status_by_client(
                BureauBalance.clean(), alpha, decimals
            ),
            "bureau_loan_status_by_client"
        )

    @classmethod
    def loan_activity_by_month(cls,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_bureau_loan_activity_by_month(BureauBalance.clean()),
            "bureau_loan_activity_by_month"
        )

    @classmethod
    def loan_activity_by_client_and_month(cls,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_bureau_loan_activity_by_client_and_month(
                BureauBalance.clean()
            ),
            "bureau_loan_activity_by_client_and_month"
        )

    @classmethod
    def mean_loan_activity(cls,
        alpha: Optional[float] = 1,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_bureau_mean_loan_activity(
                BureauBalance.clean(), alpha
            ),
            "bureau_mean_loan_activity"
        )

    @classmethod
    def mean_loan_activity_by_client(cls,
        alpha: Optional[float] = 1,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_bureau_mean_loan_activity_by_client(BureauBalance.clean()),
            "bureau_mean_loan_activity_by_client"
        )

    @classmethod
    def rle_loan_tracking_period(cls,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_rle_bureau_loan_tracking_period(
                BureauBalance.clean()
            ),
            "rle_bureau_loan_tracking_period"
        )

    @classmethod
    def rle_loan_tracking_period_by_client(cls,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_rle_bureau_loan_tracking_period_by_client(
                BureauBalance.clean()
            ),
            "rle_bureau_loan_tracking_period_by_client"
        )

    @classmethod
    def rle_loan_activity_variation(cls,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_rle_bureau_loan_feature_variation(
                BureauBalance.loan_activity_by_month(), "ACTIVITY"
            ),
            "rle_bureau_loan_activity_variation"
        )

    @classmethod
    def rle_loan_activity_by_client_variation(cls,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False,
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda:
                get_rle_bureau_loan_feature_by_client_variation(
                    BureauBalance.loan_activity_by_client_and_month(), "ACTIVE"
                ),
            "rle_bureau_loan_activity_by_client_variation"
        )

    @classmethod
    def rle_loan_status_variation(cls,
        decimals: Optional[int] = 2,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_rle_bureau_loan_feature_variation(
                BureauBalance.loan_status_by_month(decimals=decimals), "STATUS"
            ),
            "rle_bureau_loan_status_variation"
        )

    @classmethod
    def rle_loan_status_by_client_variation(cls,
        decimals: Optional[int] = 2,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False,
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda:
                get_rle_bureau_loan_feature_by_client_variation(
                    BureauBalance.loan_status_by_client_and_month(decimals=decimals), "STATUS"
                ),
            "rle_bureau_loan_status_by_client_variation"
        )

class InstallmentsPayments(HomeCreditTable):

    name = "installments_payments"

    @classmethod
    def _impute(cls, data: pd.DataFrame) -> None:
        """Impute missing values and correct outliers"""
        # Insert the aggregation counter
        # data.insert(0, "n_PREV", 1)
        
        # The basic corrective preprocessing:
        # if the installment amount is 0, we copy that of the payment.
        na_inst_case = data.AMT_INSTALMENT == 0
        data.loc[na_inst_case, "AMT_INSTALMENT"] = data[na_inst_case].AMT_PAYMENT

        # Calculate MONTHS_BALANCE based on DAYS_INSTALMENT and insert it
        gregorian_month = 365.2425 / 12
        months_balance = (-data.DAYS_INSTALMENT / gregorian_month).astype(np.uint8)
        data.insert(0, "MONTHS_BALANCE", months_balance)

    @classmethod
    def _encode(cls, data: pd.DataFrame) -> None:
        # TODO : abstraction : config dans un fichier
        ages_cols = InstallmentsPayments.cols_group("ages")
        data[ages_cols] = -data[ages_cols]

    @classmethod
    def _downcast(cls, data: pd.DataFrame, no_na_payment=False) -> None:
        # TODO : abstraction : config dans un fichier
        # Apply downcasting for various column groups
        cast_columns(data, ["SK_ID_PREV", "SK_ID_CURR"], np.uint32)
        cast_columns(data, "NUM_INSTALMENT_VERSION", np.uint8)
        cast_columns(data, ["NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT"], np.uint16)
        # Issue: float16 forbidden:
        # Unhandled type for Arrow to Parquet schema conversion: halffloat
        # cast_columns(data, ["AMT_INSTALMENT", "AMT_PAYMENT"], np.float16)
        # Precision issues : deactivate this downcast
        # cast_columns(data, ["AMT_INSTALMENT", "AMT_PAYMENT"], np.float32)
        if no_na_payment:
            cast_columns(data, "DAYS_ENTRY_PAYMENT", np.uint16)
            # cast_columns(data, "AMT_PAYMENT", np.float16)
            # cast_columns(data, "AMT_PAYMENT", np.float32)

    @classmethod
    def _get_clean_table(cls, no_na_payment=False) -> pd.DataFrame:
        # Load the raw table
        data = cls.raw()

        if no_na_payment:
            # Remove the 2,905 rows for which payment information is missing
            payment_cols = InstallmentsPayments.cols_group("payment")
            data.dropna(subset=payment_cols, inplace=True)
        
        # Cleaning pipeline
        targetize_table(data)  # Targetize (and currentize) the table if necessary
        cls._impute(data)  # Impute missing values and correct outliers
        cls._encode(data)  # Map categories
        cls._downcast(data, no_na_payment)  # Downcast data types
        cls._reset_index(data)  # Sort end reset the index and rename columns index

        return data

    @classmethod
    def clean(cls,
        no_na_payment: Optional[bool] = True,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.DataFrame:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: cls._get_clean_table(no_na_payment=no_na_payment),
            f"clean_{cls.name}{'' if no_na_payment else '_with_na_payment'}"
        )

    @classmethod
    def loan_amount(cls,
        no_na_payment: Optional[bool] = True,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.DataFrame:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_clean_installments_payments_loan_amount(
                InstallmentsPayments.clean(no_na_payment=False)
            ),
            "clean_installments_payments_loan_amount"
        )

    @classmethod
    def loaned_amount(cls,
        no_na_payment: Optional[bool] = True,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.DataFrame:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_clean_installments_payments_loaned_amount(
                InstallmentsPayments.loan_amount()
            ),
            "clean_installments_payments_loaned_amount"
        )

    @classmethod
    def last_version(cls,
        no_na_payment: Optional[bool] = True,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.DataFrame:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_clean_installments_payments_last_version(
                InstallmentsPayments.clean(no_na_payment=False)
            ),
            "clean_installments_payments_last_version"
        )


    @classmethod
    def repaid_and_dpd(cls,
        no_na_payment: Optional[bool] = True,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.DataFrame:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_clean_installments_payments_repaid_and_dpd(
                InstallmentsPayments.last_version()
            ),
            "clean_installments_payments_repaid_and_dpd"
        )

    @classmethod
    def expected_dpd_by_loan(cls,
        no_na_payment: Optional[bool] = True,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.DataFrame:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_clean_installments_payments_expected_dpd_by_loan(
                InstallmentsPayments.repaid_and_dpd(),
                InstallmentsPayments.loan_amount()
            ),
            "clean_installments_payments_expected_dpd_by_loan"
        )

    @classmethod
    def expected_dpd_by_client(cls,
        no_na_payment: Optional[bool] = True,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.DataFrame:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_clean_installments_payments_expected_dpd_by_client(
                InstallmentsPayments.repaid_and_dpd(),
                InstallmentsPayments.loaned_amount()
            ),
            "clean_installments_payments_expected_dpd_by_client"
        )


    @classmethod
    def clean_base(cls,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.DataFrame:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_clean_installments_payments_base(
                InstallmentsPayments.clean()
            ),
            "clean_installments_payments_clean_base"
        )

    @classmethod
    def by_installment(cls,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.DataFrame:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_installments_payments_by_installment(
                InstallmentsPayments.clean_base()
            ),
            "installments_payments_by_installment"
        )


class POSCashBalance(HomeCreditTable):

    name = "pos_cash_balance"

    """ Options NA included, no_NA included
    @staticmethod
    def clean_with_na() -> pd.DataFrame:
        return get_clean_pos_cash_balance_with_na()
    """

    @classmethod
    def _impute(cls, data: pd.DataFrame) -> None:
        # sourcery skip: pandas-avoid-inplace
        # -> in context, it's a very bad advice
        """Impute missing values and correct outliers"""
        data.dropna(inplace=True)

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

    @classmethod
    def rle_loan_tracking_period(cls,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_rle_pos_cash_loan_tracking_period(
                POSCashBalance.clean()
            ),
            "rle_pos_cash_loan_tracking_period"
        )


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
        cast_columns(data, "MONTHS_BALANCE", np.uint8)


    @classmethod
    def rle_loan_tracking_period(cls,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_rle_credit_card_loan_tracking_period(
                POSCashBalance.clean()
            ),
            "rle_credit_card_loan_tracking_period"
        )


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
            Bureau._to_curr_map, "bureau_to_curr_map"
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
        cast_columns(data, "SK_ID_BUREAU", np.uint32)
        cast_columns(data, "SK_ID_CURR", np.uint32)
        cast_columns(data, "CNT_CREDIT_PROLONG", np.uint8)
        cast_columns(data, "DAYS_CREDIT_UPDATE", np.int32)
        # cast_columns(data, "DAYS_CREDIT_ENDDATE", np.int32) NAs
        cast_columns(data, "DAYS_CREDIT", np.uint16)
        # cast_columns(data, "DAYS_ENDDATE_FACT", np.uint16) NAs
        cast_columns(data, "CREDIT_DAY_OVERDUE", np.uint16)
        # Downcast float32 of financial_statement is impossible : all NA (except one)

    @classmethod
    def extended_clean(cls,
        alpha: Optional[float] = 1,
        decimals: Optional[int] = 2,
        include_rle: Optional[bool] = False,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.DataFrame:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: get_extended_clean_bureau(
                Bureau.clean(),
                BureauBalance.loan_status(alpha=alpha, decimals=decimals),
                BureauBalance.loan_activity(alpha=alpha, decimals=decimals),
                (
                    BureauBalance.rle_loan_tracking_period()
                    .rename(columns={"MONTHS_BALANCE": "MONTH_support"})
                    if include_rle else None
                ),
                (
                    BureauBalance.rle_loan_status_variation(decimals=decimals)
                    .rename(columns={"STATUS": "STATUS_frame"})
                    if include_rle else None
                ),
                (
                    BureauBalance.rle_loan_activity_variation()
                    .rename(columns={"ACTIVITY": "ACTIVITY_frame"})
                    if include_rle else None
                )
            ),
            "extended_clean_bureau"
        )


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
        PreviousApplication.cast_cols_group(data, "keys", np.uint32)
        PreviousApplication.cast_cols_group(data, "last_app_flags", np.uint8)
        cast_columns(data, "SELLERPLACE_AREA", np.int32)


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
        target_map.TARGET.astype(np.int8)
        return target_map

    @classmethod
    def to_target_map(cls,
        no_cache: Optional[bool] = False,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            Application._to_target_map, "application_to_target_map"
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


class Region(HomeCreditTable):

    name = "region"

    @classmethod
    def _get_clean_table(cls) -> pd.DataFrame:
        pivot = "REGION_POPULATION_RELATIVE"
        target = "TARGET"
        commute_flags_cols = Application.cols_group("commute_flags")
        region_ratings_cols = Application.cols_group("region_ratings")
        region_cols = [target, pivot] + region_ratings_cols + commute_flags_cols

        data = Application.clean()
        region = data[region_cols]
        
        def count_if(val):
            return lambda x: (x == val).sum()

        region = region.groupby(by=pivot)

        region = region.agg({
            pivot: "count",
            "REGION_RATING_CLIENT": "first",
            target: [count_if(0), count_if(1), count_if(-1)],
            "REGION_RATING_CLIENT_W_CITY": {count_if(3), count_if(2), count_if(1)}
        } | {flag_col: "sum" for flag_col in commute_flags_cols})

        region = region.sort_values(by=(pivot, "count"), ascending=False)

        region = region.reset_index()
        region.index.name = "REG_ID"

        # TODO ce devrait être défini dans le fichier de mappage de colonnes
        # TODO c'est un cas d'application de l'abstraction de règle : tester avec l'intégration
        region.columns = pd.Index([
            ("REGION", "POPULATION"),
            ("REGION", "CLIENTS"),
            ("REGION", "RATING"),
            ("TARGET_DIST", "0"),
            ("TARGET_DIST", "1"),
            ("TARGET_DIST", "-1"),
            ("CITY_RATING_DIST", "3"),
            ("CITY_RATING_DIST", "2"),
            ("CITY_RATING_DIST", "1"),
            ("COMMUTE_FLAG_COUNTS", "C1"),
            ("COMMUTE_FLAG_COUNTS", "C2"),
            ("COMMUTE_FLAG_COUNTS", "C3"),
            ("COMMUTE_FLAG_COUNTS", "C4"),
            ("COMMUTE_FLAG_COUNTS", "C5"),
            ("COMMUTE_FLAG_COUNTS", "C6"),
        ])
        
        region.columns.names = ["REGION", "details"]
        
        return region
    
    @staticmethod
    def _to_seller_place_area_map():
        # Retrieving cleaned data
        a_reg = Application.clean()[["REGION_POPULATION_RELATIVE"]]
        pa_spa = PreviousApplication.clean()[["SELLERPLACE_AREA"]]
        reg_pop = Region.clean()[[("REGION", "POPULATION")]]

        # Merging data into an associative table
        reg_pop.columns = ["REGION_POPULATION_RELATIVE"]
        reg_pop = reg_pop.reset_index()
        reg_spa_map = pd.merge(a_reg, right=pa_spa, on="SK_ID_CURR", how="inner").drop_duplicates()
        reg_spa_map = pd.merge(reg_spa_map, right=reg_pop, on="REGION_POPULATION_RELATIVE", how="inner")

        # Cleaning and formatting the result
        reg_spa_map.columns = ["_", "SPA_ID", "REG_ID"]
        reg_spa_map = reg_spa_map[["REG_ID", "SPA_ID"]]
        reg_spa_map = reg_spa_map.sort_values(by=["REG_ID", "SPA_ID"])
        reg_spa_map = reg_spa_map.reset_index(drop=True)
        reg_spa_map.columns.name = "REG_SPA_MAP"
        
        return reg_spa_map

    @classmethod
    def to_seller_place_area_map(cls,
        no_cache: Optional[bool] = False,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.Series:
        return controlled_load(
            this_f_name(), locals().copy(),
            Region._to_seller_place_area_map, "region_to_seller_place_area_map"
        )


class SellerPlaceArea(HomeCreditTable):

    name = "seller_place_area"

    @classmethod
    def _get_clean_table(cls, normalize: bool) -> pd.DataFrame:
        normalize = True

        # Retrieve the list of categorical columns from the previous_application table.
        sales_cats = PreviousApplication.cols_group("sales_cats")

        # Retrieve the cleaned version of the previous_application table.
        previous_application = PreviousApplication.clean()

        # Select columns related to 'SELLERPLACE_AREA' and sales categories
        pa_sales = previous_application[["SELLERPLACE_AREA"] + sales_cats]
        pa_sales.columns.name = "PA_SALES"

        # Perform one-hot encoding on the selected columns
        ohe_pa_sales, _ = one_hot_encode_all_cats(pa_sales, drop_first=False, bi_index=True)

        # Group the encoded data by ('SELLERPLACE_AREA', '')
        grouped = ohe_pa_sales.groupby(by=("SELLERPLACE_AREA", ""))
        seller_place_area = grouped.agg("sum")

        # Insert a new column for the count of rows in each group
        seller_place_area.insert(0, ("PREV_APP", "size"), grouped.agg({("SELLERPLACE_AREA", ""): "size"}))

        # Sort the data based on the count of rows in descending order
        seller_place_area = seller_place_area.sort_values(by=("PREV_APP", "size"), ascending=False)

        # Set the index and column names
        seller_place_area.index.name = "SPA_ID"
        seller_place_area.columns.names = ["SELLER_PLACE_AREA", "details"]

        if normalize:
            # Normalize the data by dividing each column by the count of rows
            seller_place_area = seller_place_area.div(seller_place_area[("PREV_APP", "size")], axis=0)

        return seller_place_area

    @classmethod
    def clean(cls,
        normalize: Optional[bool] = False,
        no_cache: Optional[bool] = True,
        from_file: Optional[bool] = True,
        update_file:  Optional[bool] = False
    ) -> pd.DataFrame:
        return controlled_load(
            this_f_name(), locals().copy(),
            lambda: cls._get_clean_table(normalize=normalize),
            f"clean_{cls.name}{'_normalized' if normalize else ''}"
        )



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
    return sk_id_curr.map(Application.to_target_map()).astype(np.int8)


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
    
    target = map_curr_to_target(data.SK_ID_CURR)
    data.insert(0, "TARGET", target)


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
