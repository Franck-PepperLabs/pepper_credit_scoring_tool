import pandas as pd
import numpy as np
from pepper.cython.ufuncs import is_constant_ufunc


def is_constant(x):
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    elif isinstance(x, (list, tuple)):
        x = np.array(x)
    return is_constant_ufunc(x)
