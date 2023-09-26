# ufuncs.pyx
import numpy as np
cimport numpy as np


"""
\cython> python .\setup.py build_ext --inplace
"""

"""
def tuple_ufunc_old(np.ndarray arr):
    # DEPRECATED: old NumPy API
    cdef int i
    cdef int n = arr.shape[0]
    cdef np.ndarray[object] result = np.empty(n, dtype=object)
    
    for i in range(n):
        result[i] = (arr[i],)
    
    return result
"""

def tuple_ufunc(np.ndarray arr):
    cdef np.ndarray result = np.empty_like(arr, dtype=object)
    cdef np.ndarray arr_iter = np.ascontiguousarray(arr)

    with np.nditer(
        [arr_iter, result],
        flags=["external_loop"],
        op_flags=[["readonly"], ["writeonly"]]
    ) as it:
        for x, y in it:
            y[...] = (x,)

    return result


"""
def unique_ufunc_old(np.ndarray arr):
    # DEPRECATED: old NumPy API
    cdef int i
    cdef int n = arr.shape[0]
    cdef set unique_values = set()
    cdef np.ndarray[object] result = np.empty(n, dtype=object)
    
    for i in range(n):
        value = arr[i]
        if value not in unique_values:
            unique_values.add(value)
        result[i] = (value,)
    
    return result
"""

def unique_ufunc(np.ndarray arr):
    cdef set unique_values = set()
    cdef np.ndarray result = np.empty_like(arr, dtype=object)
    cdef np.ndarray arr_iter = np.ascontiguousarray(arr)

    with np.nditer(
        [arr_iter, result],
        flags=["external_loop"],
        op_flags=[["readonly"], ["writeonly"]]
    ) as it:
        for x, y in it:
            if x not in unique_values:
                unique_values.add(x)
            y[...] = (x,)

    return result


def is_constant_ufunc(np.ndarray arr):
    cdef int n = arr.shape[0]

    # Handle the case of an empty array
    if n == 0:
        return True

    cdef np.ndarray is_equal = arr == arr[0]
    
    return np.all(is_equal)
