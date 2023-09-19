import numpy as np
import pandas as pd


def impute_credit_card_balance_drawings(data: pd.DataFrame) -> None:
    """
    Impute missing values and perform data corrections
    for credit card balance drawings.

    Parameters
    ----------
    data : pd.DataFrame
        The input credit card balance data frame.
    """
    AMT_TOT = "AMT_DRAWINGS_CURRENT"
    CNT_TOT = "CNT_DRAWINGS_CURRENT"
    AMT_ATM = "AMT_DRAWINGS_ATM_CURRENT"
    CNT_ATM = "CNT_DRAWINGS_ATM_CURRENT"
    AMT_POS = "AMT_DRAWINGS_POS_CURRENT"
    CNT_POS = "CNT_DRAWINGS_POS_CURRENT"
    AMT_OTH = "AMT_DRAWINGS_OTHER_CURRENT"
    CNT_OTH = "CNT_DRAWINGS_OTHER_CURRENT"
    
    amt_tot = data[AMT_TOT]
    amt_sum = data[AMT_ATM] + data[AMT_POS] + data[AMT_OTH]
    amt_diff = (amt_tot - amt_sum).round(2)

    # Identify problematic cases: 749,816 NA and 7,150 non-zero diffs
    is_outlier = amt_diff != 0

    # For the 749,816 NA cases (AMT_TOT = 0): fill with 0
    is_na = amt_diff.isna()
    filled_cols = [AMT_ATM, CNT_ATM, AMT_POS, CNT_POS, AMT_OTH, CNT_OTH]
    data[filled_cols] = data[filled_cols].fillna(0)

    # For the 7,150 non-NA cases
    is_notna = is_outlier & ~is_na

    # If the amount is negative, it's an ATM transaction
    is_tot_negative = is_notna & (amt_tot < 0)
    updated_cols = [CNT_ATM, CNT_TOT, AMT_ATM]
    data.loc[is_tot_negative, updated_cols] = \
        np.array((1, 1, data[is_tot_negative][AMT_TOT]), dtype=object)

    # Otherwise, assign the amount to AMT_ATM if AMT_TOT is an integer,
    # and to AMT_POS otherwise
    is_tot_positive = is_notna & (amt_tot > 0)
    is_tot_integer = (amt_tot % 1) == 0
    is_pos_int = is_tot_positive & is_tot_integer
    is_pos_float = is_tot_positive & ~is_tot_integer
    data.loc[is_pos_int, [CNT_ATM, AMT_ATM]] = \
        np.array((1, data[is_pos_int][AMT_TOT]), dtype=object)
    data.loc[is_pos_float, [CNT_POS, AMT_POS]] = \
        np.array((1, data[is_pos_float][AMT_TOT]), dtype=object)
    data.loc[is_tot_positive, [CNT_TOT]] = 1
