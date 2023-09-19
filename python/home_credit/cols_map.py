from typing import List, Dict, Tuple, Union
import json
import os
from itertools import zip_longest


def _load_config_from_json(file_path: str) -> dict:
    """
    Load the configuration data from a JSON file.

    Parameters
    ----------
    file_path : str
        The path to the JSON file containing the configuration data.

    Returns
    -------
    dict
        A dictionary containing the configuration data.

    Example
    -------
    >>> config_data = _load_config_from_json("cols_map.json")
    """
    with open(file_path, "r", encoding='utf-8') as json_file:
        config_data = json.load(json_file)
    return config_data


# Load the global settings
_current_directory = os.path.dirname(os.path.abspath(__file__))
_cols_map_config = _load_config_from_json(
    os.path.join(_current_directory, "cols_map.json")
)


def _reload_config():
    global _cols_map_config
    _cols_map_config = _load_config_from_json(
        os.path.join(_current_directory, "cols_map.json")
    )


def get_mappings(table_name: str) -> Dict[str, List]:
    """
    Get the mappings for a specific table.

    Parameters
    ----------
    table_name : str
        The name of the table for which to retrieve the mappings.

    Returns
    -------
    Dict[str, List[str]]
        A dictionary containing the mappings for the specified table.

    Example
    -------
    >>> table_name = "credit_card_balance"
    >>> mappings = get_mappings(table_name)
    {
        "TARGET": ["ID", "TGT", ""],
        "SK_ID_PREV": ["ID", "PID", ""],
        ...
    }
    """
    return _cols_map_config["mappings"][table_name]


def get_groups(table_name: str) -> Dict[str, List[str]]:
    """
    Get the groups for a specific table.

    Parameters
    ----------
    table_name : str
        The name of the table for which to retrieve the groups.

    Returns
    -------
    Dict[str, List[str]]
        A dictionary containing the groups for the specified table.

    Example
    -------
    >>> table_name = "credit_card_balance"
    >>> groups = get_groups(table_name)
    {
        "key": ["TARGET", "SK_ID_PREV", ...],
        "payment": ["AMT_PAYMENT_CURRENT", ...],
        ...
    }
    """
    return _cols_map_config["groups"][table_name]


def get_group(table_name: str, group_name: str) -> List[str]:
    """
    Get a specific group of column names and subgroups for a given table.

    This function retrieves a specific group of column names and subgroups
    for the specified table from the predefined mappings and groups.

    Parameters
    ----------
    table_name : str
        The name of the table for which to retrieve the group.
    group_name : str
        The name of the group to retrieve.

    Returns
    -------
    List[str]
        A list containing the column names and subgroups in the specified group.

    Example
    -------
    >>> table_name = "credit_card_balance"
    >>> group_name = "payment"
    >>> payment_group = get_group(table_name, group_name)
    ["AMT_PAYMENT_CURRENT", "AMT_PAYMENT_TOTAL_CURRENT"]
    """
    return get_groups(table_name)[group_name]


def flatten_group(
    table_name: str,
    group: Union[str, List[str]]
) -> List[str]:
    """
    Flatten a group of column names and subgroups for a given table.

    This function takes a group, which can contain column names and subgroups,
    and flattens it into a single list of column names for the specified table.
    It recursively processes subgroups and handles multiplexed groups.

    Parameters
    ----------
    table_name : str
        The name of the table for which to flatten the group.
    group : str or List[str]
        The group to be flattened. It can be a string representing a single column
        or subgroup, or a list containing column names, subgroups, or both.

    Returns
    -------
    List[str]
        A list containing the flattened column names for the specified group.

    Example
    -------
    >>> table_name = "credit_card_balance"
    >>> group = ["payment", ["dpd", "status"]]
    >>> flattened_group = flatten_group(table_name, group)
    ["AMT_PAYMENT_CURRENT", "AMT_PAYMENT_TOTAL_CURRENT", "SK_DPD", "NAME_CONTRACT_STATUS", "SK_DPD_DEF"]
    
    Raises
    ------
    ValueError
        If the element type within the group is not supported.
    """
    mappings = get_mappings(table_name)
    groups = get_groups(table_name)
    
    # If the group is a string, convert it to a list for consistency
    if isinstance(group, str):
        group = [group]

    flat_group = []

    for e in group:
        if isinstance(e, str):
            if e in mappings:
                # If the element is a column name, add it to the flat_group
                flat_group.append(e)
            else:
                # If it's a subgroup, recursively flatten it
                flat_group.extend(flatten_group(table_name, groups[e]))
        elif isinstance(e, list):
            # If it's a list, it may represent multiplexed groups, so multiplex them
            flat_group.extend(multiplex(table_name, e))
        else:
            raise ValueError("Unsupported element type within the group")
    return flat_group


def multiplex(table_name: str, groups: List[List[str]]) -> List[str]:
    """
    Combine column names and subgroups from specified groups into a single list.

    This function takes a list of groups, where each group is a list of column names
    and/or subgroups, and combines them into a single list. It ensures that columns
    and subgroups within each group are alternated.

    Parameters
    ----------
    table_name : str
        The name of the table for which to multiplex the groups.
    groups : List[List[str]]
        A list of groups, where each group is a list containing column names and/or
        subgroups to be combined.

    Returns
    -------
    List[str]
        A list containing the combined column names and subgroups.

    Example
    -------
    >>> table_name = "credit_card_balance"
    >>> display(get_group(table_name, "payment"))
    >>> display(get_group(table_name, "dpd"))
    >>> display(get_group(table_name, "status"))
    >>> multiplex(table_name, ["payment", "dpd", "status"])
    ['AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT']
    ['SK_DPD', 'SK_DPD_DEF']
    ['NAME_CONTRACT_STATUS']
    ['AMT_PAYMENT_CURRENT',
    'SK_DPD',
    'NAME_CONTRACT_STATUS',
    'AMT_PAYMENT_TOTAL_CURRENT',
    'SK_DPD_DEF']
    """
    flat_groups = [flatten_group(table_name, group) for group in groups]
    multiplexed = []
    for t in zip_longest(*flat_groups):
        if t := [e for e in t if e]:
            multiplexed.extend(t)
    return multiplexed


def get_group_map(
    table_name: str,
    group: List[str]
) -> Dict[str, Tuple[str]]:
    """
    Extract a subset of column mappings for a specific table from a given group.

    This function extracts a subset of column mappings (name-to-alias tuples)
    for a specified table from a given group of column names and subgroups.

    Parameters
    ----------
    table_name : str
        The name of the table for which to extract column mappings.
    group : List[str]
        A list of column names and subgroups from which to extract mappings.

    Returns
    -------
    Dict[str, Tuple[str]]
        A dictionary containing column mappings for the specified group.

    Example:
        >>> table_name = "credit_card_balance"
        >>> group = ["AMT_BALANCE", "AMT_DRAWINGS_CURRENT"]
        >>> subset_mappings = cols_map_subset(table_name, group)
        {
            'AMT_BALANCE': ('BAL', '', 'AMT'),
            'AMT_DRAWINGS_CURRENT': ('DRW', 'TOT', 'AMT')
        }
    """
    mappings = get_mappings(table_name)
    return {name: tuple(mappings[name]) for name in group}


def get_cols_map(
    table_name: str,
    group_name: str
) -> Dict[str, Tuple[str]]:
    """
    Get column mappings for a specific group in a given table.

    This function retrieves column mappings (name-to-alias tuples) for a specific group
    in the specified table.

    Parameters
    ----------
    table_name : str
        The name of the table for which to retrieve column mappings.
    group_name : str
        The name of the group for which to retrieve mappings.

    Returns
    -------
    Dict[str, Tuple[str]]
        A dictionary containing column mappings for the specified group.

    Example
    -------
    >>> table_name = "credit_card_balance"
    >>> group_name = "payment"
    >>> payment_mappings = get_cols_map(table_name, group_name)
    {
        'AMT_PAYMENT_CURRENT': ('PYT', '', 'AMT'),
        'AMT_PAYMENT_TOTAL_CURRENT': ('PYT', 'TOT', 'AMT')
    }
    """
    group = get_group(table_name, group_name)
    flat_group = flatten_group(table_name, group)
    return get_group_map(table_name, flat_group)


"""
Early version: DEPRECATED
class CreditCardBalanceColMap:
    @staticmethod
    def get_key_cols_map():
        return {
            "TARGET": ("ID", "TGT", ""),
            "SK_ID_PREV": ("ID", "PID", ""),
            "SK_ID_CURR": ("ID", "CID", ""),
            "MONTHS_BALANCE": ("ID", "M°", "")
        }

    @staticmethod
    def get_credit_limit_cols_map():
        return {
            "AMT_CREDIT_LIMIT_ACTUAL": ("MAX", "", "AMT")
        }

    @staticmethod
    def get_balance_cols_map():
        return {
            "AMT_BALANCE": ("BAL", "", "AMT")
        }

    @staticmethod
    def get_drawings_cnt_cols_map():
        return {
            "CNT_DRAWINGS_CURRENT": ("DRW", "TOT", "CNT"),
            "CNT_DRAWINGS_ATM_CURRENT": ("DRW", "ATM", "CNT"),
            "CNT_DRAWINGS_POS_CURRENT": ("DRW", "POS", "CNT"),
            "CNT_DRAWINGS_OTHER_CURRENT": ("DRW", "OTH", "CNT")
        }

    @staticmethod
    def get_drawings_amt_cols_map():
        return {
            "AMT_DRAWINGS_CURRENT": ("DRW", "TOT", "AMT"),
            "AMT_DRAWINGS_ATM_CURRENT": ("DRW", "ATM", "AMT"),
            "AMT_DRAWINGS_POS_CURRENT": ("DRW", "POS", "AMT"),
            "AMT_DRAWINGS_OTHER_CURRENT": ("DRW", "OTH", "AMT")
        }

    @staticmethod
    def get_payment_cols_map():
        return {
            "AMT_PAYMENT_CURRENT": ("PYT", "", "AMT"),
            "AMT_PAYMENT_TOTAL_CURRENT": ("PYT", "TOT", "AMT")
        }

    @staticmethod
    def get_installment_cols_map():
        return {
            "AMT_INST_MIN_REGULARITY": ("INST", "", "AMT"),
            "CNT_INSTALMENT_MATURE_CUM": ("INST", "", "CNT")
        }

    @staticmethod
    def get_receivable_cols_map():
        return {
            "AMT_RECEIVABLE_PRINCIPAL": ("RCV", "", "AMT"),
            # "AMT_RECIVABLE": "RCV_2",
            "AMT_TOTAL_RECEIVABLE": ("RCV", "TOT", "AMT")
        }

    @staticmethod
    def get_status_cols_map():
        return {
            "NAME_CONTRACT_STATUS": ("STATUS", "", "")
        }

    @staticmethod
    def get_dpd_cols_map():
        return {
            "SK_DPD" : ("DPD", "", ""),
            "SK_DPD_DEF": ("DPD", "DEF", "")
        }

    @classmethod
    def get_balancing_cols_map(cls):
        return (
            cls.get_balance_cols_map() |
            {
                "AMT_DRAWINGS_CURRENT": ("DRW", "TOT", "AMT"),
            } |
            cls.get_receivable_cols_map() |
            cls.get_payment_cols_map()
        )

    @classmethod
    def get_drawings_amt_cnt_couples_cols_map(cls):
        cols_map = {}
        amt_cols_map = cls.get_drawings_amt_cols_map()
        cnt_cols_map = cls.get_drawings_cnt_cols_map()
        for amt, cnt in zip(amt_cols_map.keys(), cnt_cols_map.keys()):
            cols_map[amt] = amt_cols_map[amt]
            cols_map[cnt] = cnt_cols_map[cnt]
        return cols_map

    @classmethod
    def get_all_cols_map(cls):
        return (
            cls.get_key_cols_map() |
            cls.get_status_cols_map() |
            cls.get_credit_limit_cols_map() |
            cls.get_balance_cols_map() |
            cls.get_drawings_amt_cols_map() |
            cls.get_drawings_cnt_cols_map() |
            cls.get_receivable_cols_map() |
            cls.get_installment_cols_map() |
            cls.get_payment_cols_map() |
            cls.get_dpd_cols_map()
        )


    @staticmethod
    def _get_cols_group(col_map_provider, shorten=False):
        cols_map = col_map_provider()
        return list(cols_map.values() if shorten else cols_map.keys())

    @classmethod
    def get_key_cols(cls, shorten=False):
        return cls._get_cols_group(cls.get_key_cols_map, shorten)

    @classmethod
    def get_drawings_cnt_cols(cls, shorten=False):
        return cls._get_cols_group(cls.get_drawings_cnt_cols_map, shorten)

    @classmethod
    def get_drawings_amt_cols(cls, shorten=False):
        return cls._get_cols_group(cls.get_drawings_amt_cols_map, shorten)

    @classmethod
    def get_drawings_amt_cnt_couples_cols(cls, shorten=False):
        return cls._get_cols_group(cls.get_drawings_amt_cnt_couples_cols_map, shorten)

    @classmethod
    def get_payment_cols(cls, shorten=False):
        return cls._get_cols_group(cls.get_payment_cols_map, shorten)

    @classmethod
    def get_balancing_cols(cls, shorten=False):
        return cls._get_cols_group(cls.get_balancing_cols_map, shorten)

    @classmethod
    def get_all_cols(cls, shorten=False):
        return cls._get_cols_group(cls.get_all_cols_map, shorten)
"""