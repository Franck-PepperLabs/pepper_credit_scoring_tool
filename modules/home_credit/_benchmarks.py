import timeit

import pandas as pd

from home_credit.load import get_target, get_previous_application
from home_credit.merge import targetize


def targetize_v1(sk_id_curr: pd.Series) -> pd.Series:
    return pd.merge(sk_id_curr, get_target(), how="left", on="SK_ID_CURR").TARGET


def bench_targetize():
    sk_id_curr = get_previous_application().SK_ID_CURR

    tgt_1 = targetize_v1(sk_id_curr)
    tgt_2 = targetize(sk_id_curr)

    # Ident
    print("Id:", (tgt_1 == tgt_2).all())

    # Perf
    n_iter = 100
    print(f"targetize_v1: {timeit.timeit(lambda: targetize_v1(sk_id_curr), number=n_iter)} s")
    print(f"targetize_v2: {timeit.timeit(lambda: targetize(sk_id_curr), number=n_iter)} s")

    """
    Id: True
    targetize_v1: 34.859060300004785 s
    targetize_v2: 6.256639500003075 s
    """
