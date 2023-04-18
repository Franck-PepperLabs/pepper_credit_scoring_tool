import pandas as pd

from pepper.np_utils import subindex as npsi


def subindex(s, sorted=False):
    # TODO comparer les deux approches avec timeit pour choisir laquelle conserver
    """df = s.reset_index()
    df.insert(2, "subindex", npsi(s.values))
    df.set_index("index", inplace=True)
    df.index.name = None
    return df"""
    s_sub = pd.Series(
        npsi(s.values, sorted),
        index=s.index, name="subindex"
    )
    return pd.concat([s, s_sub], axis=1)
