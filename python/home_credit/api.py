from typing import Union, Tuple, List

import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd

from home_credit.load import get_table, get_table_loaders_dict


def _parse_range_args(*args: Union[Tuple[int], Tuple[int, int]]) -> Tuple[int, int]:
    """
    Parse the arguments for a start, stop *args section.
    Variable-length arguments representing the start and stop values of the range.

    - If no arguments are provided, it creates an infinite range ('stop'=-1) starting at 0.
    - If a single argument 'stop' is provided, it defines a range from 0 to 'stop' (not included).
    - If two arguments 'start' and 'stop' are provided, it defines a range from 'start' to 'stop' (not included).

    Parameters
    ----------
    *args : Union[Tuple[int], Tuple[int, int]]
        Variable-length arguments representing the start and stop values.

    Returns
    -------
    Tuple[int, int]
        A tuple containing the start and stop values for the range.

    Raises
    ------
    ValueError
        If the number of arguments is not 0, 1, or 2.
    ValueError
        If 'start' or 'stop' is not an integer.

    Notes
    -----
    If 'start' and 'stop' are both negative, they are considered relative to the end of the table.
    If 'start' modulo the table size is greater than 'stop' modulo the table size, it corresponds to an empty selection.
    """
    if not args:
        start, stop = 0, -1
    elif len(args) == 1:
        start, stop = 0, args[0]
    elif len(args) == 2:
        start, stop = args[0], args[1]
    else:
        raise ValueError("Expected 0, 1, or 2 arguments")

    if not isinstance(start, int) or not isinstance(stop, int):
        raise ValueError("Both start and stop must be integers")

    return start, stop


def get_table_range(
    table_name: str,
    *args: Union[Tuple[int], Tuple[int, int]]
) -> pd.DataFrame:
    """
    Get a specified range of rows from a table.

    Parameters
    ----------
    table_name : str
        The name of the table to retrieve data from.

    *args : Union[Tuple[int], Tuple[int, int]]
        Variable-length arguments representing the start and stop values for the range.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the selected range of rows from the table.

    Raises
    ------
    ValueError
        If the specified table_name is not found in the available tables.

    Notes
    -----
    This function retrieves the specified range of rows from the specified table.
    The range is defined using the *args parameter, which allows you to specify
    either a single 'stop' value to get rows from 0 to 'stop' (not included), or
    two values 'start' and 'stop' to get rows from 'start' to 'stop' (not included).
    If 'start' and 'stop' are both negative, they are considered relative to the end
    of the table. If 'start' modulo the table size is greater than 'stop' modulo
    the table size, it corresponds to an empty selection.

    Example
    -------
    To retrieve the first 10 rows of a table:
    >>> get_table_range("my_table", 10)

    To retrieve rows 5 to 15 (not included) of a table:
    >>> get_table_range("my_table", 5, 15)
    """
    # Load the table from the (thread-safe) cache
    table = get_table(table_name)

    # Parse the range arguments using _parse_range_args
    start, stop = _parse_range_args(*args)
    print(start, stop)

    logging.info(f"api.get_table_range({table_name}, {start}, {stop})")

    # Return the selected range of rows using iloc
    return table.iloc[start:stop]


def get_table_names() -> List[str]:
    """
    Get a list of available table names.

    Returns
    -------
    List[str]
        A list of table names.
    """
    return list(get_table_loaders_dict().keys())

