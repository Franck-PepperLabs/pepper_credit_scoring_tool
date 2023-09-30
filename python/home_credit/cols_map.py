from typing import List, Dict, Tuple, Union, Any
import os, json
from itertools import zip_longest, product


""" Abstract rules
"""


def concretize_set(param_def: Union[list, dict]) -> list:
    """
    Concretize a parameter definition into a list of values.

    Parameters
    ----------
    param_def : list or dict
        The parameter definition to concretize.
        It can be a list of values or a dictionary with a "range" key.

    Returns
    -------
    list
        A list of concrete values.

    Examples
    --------
    >>> concretize_set([1, 2, 3])
    [1, 2, 3]

    >>> concretize_set({"range": [1, 4]})
    [1, 2, 3]
    """
    if isinstance(param_def, list):
        return param_def
    elif isinstance(param_def, dict):
        return range(*param_def["range"])


def format_key(key_pattern: str, format: dict) -> str:
    """
    Format a key based on a pattern and a dictionary of parameter values.

    Parameters
    ----------
    key_pattern : str
        The pattern for formatting keys.
        Placeholders for parameters are enclosed in curly braces.
    format : dict
        A dictionary of parameter names and their corresponding values.

    Returns
    -------
    str
        The formatted key.

    Examples
    --------
    >>> format_key("{x}_{y}", {'x': 'A', 'y': 1})
    'A_1'
    """
    params_dict = dict(format)
    formatted_key = key_pattern
    for param, value in params_dict.items():
        formatted_key = formatted_key.replace(f"{{{param}}}", f"{value}")
    return formatted_key
    # TODO Why some cases have issues with this version ?
    # return key_pattern.format(**format)


def format_value(
    value_pattern: Union[str, List[str]],
    format: Dict[str, Any]
) -> Union[str, List[str]]:
    """
    Format a single value or a list of values based on a pattern
    and a dictionary of parameter values.

    Parameters
    ----------
    value_pattern : str or List[str]
        The pattern(s) for formatting values.
        If a string, it represents a single value pattern.
        If a list of strings, each string represents a value pattern.
    format : dict
        A dictionary of parameter names and their corresponding values.

    Returns
    -------
    str or List[str]
        The formatted value or a list of formatted values.

    Examples
    --------
    >>> format_value("Value_{x}", {'x': 'A'})
    'Value_A'

    >>> format_value(["Value_{x}", "Number_{y}"], {'x': 'A', 'y': 1})
    ['Value_A', 'Number_1']
    """
    if isinstance(value_pattern, str):
        return format_key(value_pattern, format)
    return [
        format_key(pat, format)
        for pat in value_pattern
    ]


def format_keys(
    key_pattern: str,
    formats: List[List[Tuple[str, Union[str, Any]]]]
) -> List[str]:
    """
    Format keys based on a pattern and parameter combinations.

    Parameters
    ----------
    key_pattern : str
        The pattern for formatting keys.
        Placeholders for parameters are enclosed in curly braces.
    formats : List[List[Tuple[str, Union[str, Any]]]]
        A list of parameter combinations, where each combination is
        a list of tuples (param_name, param_value).

    Returns
    -------
    List[str]
        A list of formatted keys.

    Examples
    --------
    >>> combined_params = [[('x', 'A'), ('x', 'B')], [('y', 1), ('y', 2), ('y', 3)]]
    >>> format_keys("{x}_{y}", combined_params)
    ['A_1', 'A_2', 'A_3', 'B_1', 'B_2', 'B_3']
    """
    return [
        format_key(key_pattern, format)
        for format in product(*formats)
    ]


def format_values(
    value_pattern: Union[str, List[str]],
    formats: List[List[Tuple[str, Any]]]
) -> Union[str, List[List[str]]]:
    """
    Format a list of values based on a pattern and a list of lists of parameter values.

    Parameters
    ----------
    value_pattern : str or List[str]
        The pattern(s) for formatting values.
        If a string, it represents a single value pattern.
        If a list of strings, each string represents a value pattern.
    formats : List[list]
        A list of lists, where each inner list contains parameter names
        and their corresponding values.

    Returns
    -------
    str or List[List[str]]
        The formatted values. If `value_pattern` is a string,
        the return type is `str`.
        If `value_pattern` is a list of strings,
        the return type is `List[List[str]]`.

    Examples
    --------
    >>> value_pattern = "Value_{x}_{y}"
    >>> formats = [[('x', 'A'), ('y', 1)], [('x', 'B'), ('y', 2)]]
    >>> format_values(value_pattern, formats)
    ['Value_A_1', 'Value_B_2']

    >>> value_pattern = ["Value_{x}", "Number_{y}"]
    >>> formats = [[('x', 'A'), ('y', 1)], [('x', 'B'), ('y', 2)]]
    >>> format_values(value_pattern, formats)
    [['Value_A', 'Number_1'], ['Value_B', 'Number_2']]
    """
    return [
        format_value(value_pattern, format)
        for format in product(*formats)
    ]


def combine_params(
    pat_params: Dict[str, Union[list, dict]]
) -> List[List[Tuple[str, Any]]]:
    """
    Combine parameter definitions into a list of parameter-value pairs
    for each parameter.

    Parameters
    ----------
    pat_params : dict
        A dictionary where keys are parameter names, and values are either
        lists representing parameter value sets or dictionaries with a "range"
        key defining a range of values.

    Returns
    -------
    list
        A list of lists, where each inner list contains tuples representing
        parameter name-value pairs.

    Example
    -------
    >>> pat_params = {'x': [1, 2, 3], 'y': {'range': [0, 5]}}
    >>> combine_params(pat_params)
    [[('x', 1), ('x', 2), ('x', 3)],
     [('y', 0), ('y', 1), ('y', 2), ('y', 3), ('y', 4)]]
    """
    return [
        [(param, value) for value in concretize_set(param_def)]
        for param, param_def in pat_params.items()
    ]


def concretize_mapping(
    key_pat: str,
    val_pat: Union[str, List[str]],
    pat_params: Dict[str, Union[list, dict]]
) -> Dict[str, Union[str, List[str]]]:
    """
    Concretize an abstract mapping by generating
    a dictionary of concrete key-value pairs.

    Parameters
    ----------
    key_pat : str
        The key pattern specifying the format of the keys.
    val_pat : Union[str, List[str]]
        The value pattern specifying the format of the values.
        It can be a string pattern or
        a list of string patterns.
    pat_params : dict
        A dictionary where keys are parameter names, and values are either
        lists representing parameter value sets or dictionaries
        with a "range" key defining a range of values.

    Returns
    -------
    dict
        A dictionary where keys are concrete keys generated based
        on the key pattern and parameter values, and values are concrete values
        generated based on the value pattern and parameter values.

    Example
    -------
    >>> key_pat = "{x}_{y}"
    >>> val_pat = "Value_{x}_{y}"
    >>> pat_params = {'x': [1, 2], 'y': {'range': [0, 2]}}
    >>> concretize_mapping(key_pat, val_pat, pat_params)
    {'1_0': 'Value_1_0', '1_1': 'Value_1_1', '2_0': 'Value_2_0', '2_1': 'Value_2_1'}
    
    >>> key_pat = "{x}_{y}"
    >>> val_pat = ["Value_{x}_{y}", "{y}_{x}"]
    >>> pat_params = {'x': [1, 2], 'y': {'range': [0, 2]}}
    >>> concretize_mapping(key_pat, val_pat, pat_params)
    {'1_0': ['Value_1_0', '0_1'], '1_1': ['Value_1_1', '1_1'],
     '2_0': ['Value_2_0', '0_2'], '2_1': ['Value_2_1', '1_2']}
    """
    formats = combine_params(pat_params)
    return dict(zip(
        format_keys(key_pat, formats),
        format_values(val_pat, formats)
    ))


def concretize_group(
    val_pat: Union[str, List[str]],
    pat_params: dict
) -> Union[str, List[str]]:
    """
    Concretize an abstract group by generating concrete values based
    on parameter values.

    Parameters
    ----------
    val_pat : Union[str, List[str]]
        The value pattern specifying the format of the values.
        It can be a string pattern or a list of string patterns.
    pat_params : dict
        A dictionary where keys are parameter names, and values are
        either lists representing parameter value sets or dictionaries with
        a "range" key defining a range of values.

    Returns
    -------
    Union[str, List[str]]
        Concrete values generated based on the value pattern and parameter values.
        If val_pat is a list, a list of concrete values is returned.

    Example
    -------
    >>> val_pat = "Value_{x}_{y}"
    >>> pat_params = {'x': [1, 2], 'y': {'range': [0, 2]}}
    >>> concretize_group(val_pat, pat_params)
    ['Value_1_0', 'Value_1_1', 'Value_2_0', 'Value_2_1']
    
    >>> val_pat = ["Value_{x}_{y}", "{y}_{x}"]
    >>> pat_params = {'x': [1, 2], 'y': {'range': [0, 2]}}
    >>> concretize_group(val_pat, pat_params)
    [['Value_1_0', '0_1'],
     ['Value_1_1', '1_1'],
     ['Value_2_0', '0_2'],
     ['Value_2_1', '1_2']]
    """
    formats = combine_params(pat_params)
    return format_values(val_pat, formats)


def concretize_abstract_groups(config: dict) -> None:
    """
    Concretize abstract groups in a configuration dictionary.

    Parameters
    ----------
    config : dict
        The configuration dictionary containing abstract groups.

    Returns
    -------
    None

    Description
    -----------
    This function processes the abstract groups in the configuration dictionary
    and replaces them with concrete values based on parameter values.
    It modifies the input `config` dictionary in-place.

    The abstract groups are expected to have the following structure:
    {
        "var": value_pattern,  # A string pattern or list of patterns
        "par": {
            parameter1: parameter1_values,
            parameter2: parameter2_values,
            ...
        }
    }
    
    This function iterates through all abstract groups within each table's groups
    and replaces them with concrete values based on parameter combinations.

    Example
    -------
    >>> config = {
    ...     "groups": {
    ...         "table1": {
    ...             "group1": {
    ...                 "var": "Value_{x}_{y}",
    ...                 "par": {"x": [1, 2], "y": {"range": [0, 2]}}
    ...             }
    ...         }
    ...     }
    ... }
    >>> concretize_abstract_groups(config)
    >>> print(config)
    {
        "groups": {
            "table1": {
                "group1": [
                    "Value_1_0", "Value_1_1", "Value_2_0", "Value_2_1"
                ]
            }
        }
    }
    """
    groups = config["groups"]
    for table_groups in groups.values():
        for k, v in table_groups.items():
            if isinstance(v, dict):
                concrete_group = concretize_group(v["var"], v["par"])
                table_groups[k] = concrete_group


def concretize_abstract_mappings(config: dict) -> None:
    """
    Concretize abstract mappings in a configuration dictionary.

    Parameters
    ----------
    config : dict
        The configuration dictionary containing abstract mappings.

    Returns
    -------
    None

    Description
    -----------
    This function processes the abstract mappings in the configuration
    dictionary and replaces them with concrete key-value pairs based
    on parameter values. It modifies the input `config` dictionary
    in-place.

    The abstract mappings are expected to have the following structure:
    {
        "map": key_pattern,  # A string pattern
        "par": {
            parameter1: parameter1_values,
            parameter2: parameter2_values,
            ...
        }
    }
    
    This function iterates through all abstract mappings within each table's
    mappings and replaces them with concrete key-value pairs based on parameter
    combinations.

    Example
    -------
    >>> config = {
    ...     "mappings": {
    ...         "table1": {
    ...             "mapping1": {
    ...                 "map": "{x}_{y}",
    ...                 "par": {"x": [1, 2], "y": {"range": [0, 2]}}
    ...             }
    ...         }
    ...     }
    ... }
    >>> concretize_abstract_mappings(config)
    >>> print(config)
    {
        "mappings": {
            "table1": {
                "VAR_1_0": ["VAR", "1", "0"],
                "VAR_1_1": ["VAR", "1", "1"],
                "VAR_2_0": ["VAR", "2", "0"],
                "VAR_2_1": ["VAR", "2", "1"]
            }
        }
    }
    """
    mappings = config["mappings"]
    for table_mappings in mappings.values():
        updates = {}
        drops = []
        for k, v in table_mappings.items():
            if isinstance(v, dict):
                concrete_mapping = concretize_mapping(k, v["map"], v["par"])
                updates |= concrete_mapping
                drops.append(k)
        for k in drops:
            del table_mappings[k]
        table_mappings.update(updates)


""" Load config
"""


def _load_config_from_json(file_path: str) -> dict:
    """
    Load the configuration data from a JSON file
    and concretize abstract mappings and groups.

    Parameters
    ----------
    file_path : str
        The path to the JSON file containing the configuration data.

    Returns
    -------
    dict
        A dictionary containing the configuration data
        with concrete mappings and groups.

    Example
    -------
    >>> config_data = _load_config_from_json("cols_map.json")
    """
    with open(file_path, "r", encoding='utf-8') as json_file:
        config = json.load(json_file)
    concretize_abstract_groups(config)
    concretize_abstract_mappings(config)
    return config


def _load_config_from_json_deprecated(file_path: str) -> dict:
    """ DEPRECATED
    
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


""" Getters
"""


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


""" Complex ops
"""


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


""" Interface
"""

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
