from home_credit.load import load_raw_table
from home_credit.merge import targetize
from home_credit.feat_eng import negate_numerical_data
from IPython.display import display
from pepper.utils import print_subtitle
import pandas as pd

from home_credit.cols_map import CCBColMap

def load_credit_card_balance():
    data = load_raw_table("credit_card_balance")
    targetize(data)
    negate_numerical_data(data.MONTHS_BALANCE)
    return data

def get_credit_card_balance_drawings(with_key_cols=True, shorten=True):
    data = load_credit_card_balance()
    cols = CCBColMap.get_key_cols() if with_key_cols else []
    cols += CCBColMap.get_drawings_amt_cnt_couples_cols()
    data = data[cols]
    if shorten:
        short_cols = CCBColMap.get_key_cols(True) if with_key_cols else []
        short_cols += CCBColMap.get_drawings_amt_cnt_couples_cols(True)
        data.columns = short_cols
    return data

def get_credit_card_balance_payment(with_key_cols=True, shorten=True):
    data = load_credit_card_balance()
    cols = CCBColMap.get_key_cols() if with_key_cols else []
    cols += CCBColMap.get_payment_cols()
    data = data[cols]
    if shorten:
        short_cols = CCBColMap.get_key_cols(True) if with_key_cols else []
        short_cols += CCBColMap.get_payment_cols(True)
        data.columns = short_cols
    return data

def get_credit_card_balance_payment_balance(with_key_cols=True, shorten=True):
    data = load_credit_card_balance()
    cols = CCBColMap.get_key_cols() if with_key_cols else []
    cols += CCBColMap.get_balancing_cols()
    data = data[cols]
    if shorten:
        short_cols = CCBColMap.get_key_cols(True) if with_key_cols else []
        short_cols += CCBColMap.get_balancing_cols(True)
        data.columns = pd.Index(short_cols)
    return data

def get_raw_loan(data: pd.DataFrame, loan_id, cols_map=None):
    loan = data[data.SK_ID_PREV == loan_id].copy()
    MB_col = "MONTHS_BALANCE"
    loan.sort_values(by=MB_col, ascending=False, inplace=True)
    if cols_map:
        loan = loan[cols_map.keys()]
        loan.columns = cols_map.values()
        MB_col = cols_map[MB_col]
    loan.index = list(loan[[MB_col]].iloc[:, 0])
    return loan


def analyze_loan(balance, loan_id):
    # balance = get_credit_card_balance_payment_balance()
    # TODO : mettre en cache
    loan = balance[balance.PID == loan_id].sort_values(by="M°", ascending=False).copy()
    loan["BAL_diff"] = loan.BAL.diff()
    loan["BAL-RCV"] = loan.BAL - loan.RCV_TOT
    loan["IF"] = loan.RCV_TOT-loan.RCV
    loan["REF_pre"] = loan.RCV_TOT.shift(1)
    loan["rate"] = loan.IF / loan.REF_pre
    loan["D-P"] = loan.DRAW-loan.PYT_TOT
    return loan


def report_loan_analysis(balance, loan_id):
    loan = analyze_loan(balance, loan_id)
    print_subtitle(f"TGT: {loan.iloc[0, 0]} | PID: {loan.iloc[0, 1]} | CID: {loan.iloc[0, 2]}")
    loan = loan.drop(columns=loan.columns[:3])
    display(loan)


def aggregate_loans():
    balance = get_credit_card_balance_payment_balance()
    balance.columns = [
        "TGT", "PID", "CID", "M°", "BAL",
        "DRW", "RCV", "RCV_TOT", "PYT", "PYT_TOT"
    ]
    balance = balance.sort_values(by=["PID", "CID", "M°"], ascending=False)
    return (
        balance[balance.columns[1:]]
        .groupby(by=list(balance.columns[1:3]))
        .agg({
            "M°": "count",
            "BAL": "sum",
            "DRW": "sum",
            "RCV": "sum",
            "RCV_TOT": "sum",
            "PYT": "sum",
            "PYT_TOT": "sum"
        })
        .rename(columns={
            "M°": "# M°",
            "BAL": "Σ BAL",
            "DRW": "Σ DRAW",
            "RCV": "Σ RCV",
            "RCV_TOT": "Σ RCV_TOT",
            "PYT": "Σ PYT",
            "PYT_TOT": "Σ PYT_TOT"
        })
    )


def _get_loans_boolean_index_predicates(loans):
    is_null_s_bal = loans["Σ BAL"] == 0
    is_null_s_draw = loans["Σ DRAW"] == 0
    is_null_s_pyt = loans["Σ PYT"] == 0
    is_null_s_pyt_tot = loans["Σ PYT_TOT"] == 0
    return is_null_s_bal, is_null_s_draw, is_null_s_pyt, is_null_s_pyt_tot

def get_null_loans_boolean_index(loans):
    is_null_s_bal, is_null_s_draw, is_null_s_pyt, is_null_s_pyt_tot = \
        _get_loans_boolean_index_predicates(loans)
    return is_null_s_bal & is_null_s_draw & is_null_s_pyt & is_null_s_pyt_tot

def get_not_balanced_loans_boolean_index(loans):
    is_null_s_bal, is_null_s_draw, is_null_s_pyt, is_null_s_pyt_tot = \
        _get_loans_boolean_index_predicates(loans)
    return is_null_s_bal & ~(is_null_s_draw & is_null_s_pyt & is_null_s_pyt_tot)


def get_loan_index(loan_boolean_index):
    return list(loan_boolean_index[loan_boolean_index].index.get_level_values(0))


def view_filtered_raw_loan(data: pd.DataFrame, loan_id):
    loan = data[data.SK_ID_PREV == loan_id].copy()
    loan.sort_values(by="MONTHS_BALANCE", ascending=False, inplace=True)
    print_subtitle(
        f"TARGET: {loan.iloc[0, 0]} | "
        f"SK_ID_PREV: {loan.iloc[0, 1]} | "
        f"SK_ID_CURR: {loan.iloc[0, 2]}"
    )
    loan.set_index("MONTHS_BALANCE", inplace=True)
    loan.drop(columns=loan.columns[:3], inplace=True)
    tr_loan = loan.T
    is_not_null_row = (~(tr_loan.isna() | (tr_loan == 0))).any(axis=1)
    display(tr_loan[is_not_null_row])
