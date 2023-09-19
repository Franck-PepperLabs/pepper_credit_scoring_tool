from typing import Union, Optional, List, Tuple, Dict, Type

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import timeit
from IPython.display import display

from pepper.utils import discrete_stats, set_plot_title, save_and_show


""" Discrete stats
"""


def table_discrete_stats(
    table_name: str,
    table: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate discrete statistics for a table and add the table name as a column.

    Parameters
    ----------
    table_name : str
        The name of the table.
    table : pd.DataFrame
        The DataFrame for which to compute statistics.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing discrete statistics for the input table with an additional
        'table_name' column.

    Examples
    --------
    >>> import pandas as pd
    >>> from pepper.utils import discrete_stats
    >>> data = pd.DataFrame({'A': [1, 2, 3, np.nan], 'B': ['cat', 'dog', 'dog', 'dog']})
    >>> table_discrete_stats('test', data)
    table_name   col  n  n_u  n_na  Fill rate  Diversity rate   dtypes
    0           test    A  3    3     1      0.750000         0.750000  float64
    1           test    B  3    2     0      1.000000         0.666667   object
    """
    stats = discrete_stats(table)
    stats.reset_index(inplace=True, names="col")
    stats.insert(loc=0, column="table_name", value=table_name)
    return stats


def db_discrete_stats(
    table_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Calculate discrete statistics for multiple tables in a database.

    Parameters
    ----------
    table_dict : Dict[str, pd.DataFrame]
        A dictionary where keys are table names and values are DataFrames
        representing the tables.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing discrete statistics for all tables in the database.

    Examples
    --------
    >>> import pandas as pd
    >>> from pepper.utils import discrete_stats
    >>> data1 = pd.DataFrame({'A': [1, 2, 3, np.nan], 'B': ['cat', 'dog', 'dog', 'dog']})
    >>> data2 = pd.DataFrame({'X': [4, 5, 6], 'Y': [1.2, 3.4, 5.6]})
    >>> table_dict = {'table1': data1, 'table2': data2}
    >>> db_discrete_stats(table_dict)
    table_name   col  n  n_u  n_na  Fill rate  Diversity rate   dtypes
    0       table1    A  3    3     1      0.750000         0.750000  float64
    1       table1    B  3    2     0      1.000000         0.666667   object
    2       table2    X  3    3     0      1.000000         1.000000  int64
    3       table2    Y  3    3     0      1.000000         1.000000  float64
    """
    return pd.concat([
        table_discrete_stats(table_name, table)
        for table_name, table in table_dict.items()
    ], axis=0, ignore_index=True)


""" Indexing, casting
"""


def set_pk_index(
    data: pd.DataFrame,
    pk_columns: Union[str, List[str]],
    pk_name: Optional[str] = None
):
    """
    Set the primary key index of the data table.

    Parameters
    ----------
    data : pd.DataFrame
        The data table.
    pk_columns : Union[str, List[str]]
        The primary key column(s). If a single string, it will be used as the
        primary key index directly. If a list, the primary key index will be a
        tuple of the values in the list.
    pk_name : str, optional
        The name of the primary key index. If not provided, the primary key
        index name will be the concatenation of the primary key column names in
        parentheses if pk_columns is a list, or the single column name if
        pk_columns is a string.

    Returns
    -------
    None
        Inplace.
    """
    # If the primary key has only one column, set it as the index directly
    if isinstance(pk_columns, str):
        data.set_index(pk_columns, drop=True, inplace=True)
    elif len(pk_columns) == 1:
        data.set_index(pk_columns[0], drop=True, inplace=True)
    else:
        # If the primary key name is not provided, create it from the
        # primary key column names
        if pk_name is None:
            pk_name = "(" + ", ".join(pk_columns) + ")"
            # bad ? : pk_name = tuple(pk_columns)

        # Create a Series with the primary key values as tuples
        pk = pd.Series(
            list(zip(*[data[col] for col in pk_columns]))
        ).rename(pk_name)

        # Set the primary key Series as the index of the data table
        data.set_index(pk, inplace=True)
        data.drop(columns=pk_columns, inplace=True)


def cast_columns(
    data: pd.DataFrame,
    cols: Union[str, List[str]],
    dtype: Type
):
    """
    Cast the specified columns in the dataframe to the specified dtype.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe.
    cols : Union[str, list]
        The column name(s) to cast. If a single string, it will be used as the
        column name to cast. If a list, all the columns in the list will be
        cast.
    dtype : Type
        The type to cast the columns to (e.g. int, float, etc).

    Returns
    -------
    None
        Inplace.

    Example
    -------
        # Cast the weight, length, height and width columns to float
        >>> cast_columns(products,
        >>>     ['weight_g', 'length_cm', 'height_cm', 'width_cm']
        >>> , float)
        >>>
        >>> # Cast the 'price' column to float
        >>> cast_columns(products, 'price', float)
        # Cast the 'shipping_limit_date' column to 'datetime64[ns]'
        >>> cast_columns(order_items, 'shipping_limit_date', 'datetime64[ns]')
        # Cast the 'price' and 'freight_value' columns to float
        >>> cast_columns(order_items, ['price', 'freight_value'], float)
    """
    if isinstance(cols, str):
        cols = [cols]

    # Check if the s Series can be converted to the target type
    def is_castable(s: pd.Series, dtype: Type) -> bool:
        try:
            s.astype(dtype)
            return True
        except ValueError:
            return False

    for col in cols:
        if not is_castable(data[col], dtype):
            raise ValueError(
                f"Cannot cast column '{col}' to type '{dtype}'. "
                f"Some values are not castable."
            )

    # Proceed the cast
    data[cols] = data[cols].astype(dtype)  #, copy=False)


""" Relationships, arities
"""


def count_of_objets_A_by_objet_B(
    table: pd.DataFrame,
    col_A: str, col_B: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Counts the number and frequency of values of column `col_A` in `table`
    by values of column `col_B`.
    
    Parameters
    ----------
    table : pd.DataFrame
        Input data.
    col_A : str
        Name of the column to count values in `table`.
    col_B : str
        Name of the column to group by.
    
    Returns
    -------
    count_freq : pd.DataFrame
        Count and frequency of `col_A` values by `col_B` values.
    grouped : pd.DataFrame
        Result of the grouping.
    """
    grouped = table[[col_A, col_B]].groupby(by=col_B).count()
    count = grouped[col_A].value_counts().rename("count")
    freq = grouped[col_A].value_counts(normalize=True).rename("freq")
    count_freq = pd.concat([count, freq], axis=1)
    count_freq.index.name = f"{col_A} by {col_B}"
    count_freq["count"] = count_freq["count"].astype(int)
    return count_freq, grouped


def out_of_intersection_v1(
    table_A: pd.DataFrame,
    table_B: pd.DataFrame,
    pk_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the primary keys that are in one table but not the other.

    DEPRECATED [Bad perf] Use `out_of_intersection` instead

    Parameters
    ----------
    table_A : pd.DataFrame
        First table.
    table_B : pd.DataFrame
        Second table.
    pk_name : str
        Name of the primary key column.

    Returns
    -------
    pk_A : np.ndarray
        Primary keys of `table_A`.
    pk_B : np.ndarray
        Primary keys of `table_B`.
    pk_A_not_B : np.ndarray
        Primary keys of `table_A` that are not in `table_B`.
    pk_B_not_A : np.ndarray
        Primary keys of `table_B` that are not in `table_A`.
    """
    pk_A = table_A[pk_name].unique()
    pk_B = table_B[pk_name].unique()
    pk_A_not_B = np.array(list(set(pk_A) - set(pk_B)))
    pk_B_not_A = np.array(list(set(pk_B) - set(pk_A)))
    return pk_A, pk_B, pk_A_not_B, pk_B_not_A


def print_out_of_intersection(
    table_A: pd.DataFrame,
    table_B: pd.DataFrame,
    pk_name: str
) -> None:
    """
    Print the number and percentage of primary keys that are in one table
    but not the other.

    Parameters
    ----------
    table_A : pd.DataFrame)
        First table.
    table_B : pd.DataFrame
        Second table.
    pk_name : str
        Name of the primary key column.
    """
    (
        pk_A, pk_B, pk_A_not_B, pk_B_not_A
    ) = out_of_intersection(table_A, table_B, pk_name)
    name_A = table_A.columns.name
    name_B = table_B.columns.name
    print(f"|{name_A}.{pk_name}| : {len(pk_A)}")
    print(f"|{name_B}.{pk_name}| : {len(pk_B)}")
    print(
        f"|{name_A}.{pk_name} \\ {name_B}.{pk_name}| : {len(pk_A_not_B)}",
        f"({str(round(100 * len(pk_A_not_B) / len(pk_A), 3))}%)",
    )
    print(
        f"|{name_B}.{pk_name} \\ {name_A}.{pk_name}| : {len(pk_B_not_A)}",
        f"({str(round(100 * len(pk_B_not_A) / len(pk_B), 3))}%)",
    )


def display_relation_arities(
    table_A: pd.DataFrame, pk_A: str,
    table_B: pd.DataFrame, fk_B: str,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float, float, float]:
    """
    Compute and display statistics about the relation between two tables.

    Parameters
    ----------
    table_A : pd.DataFrame
        First table.
    table_B : pd.DataFrame
        Second table.
    pk_A : str
        Primary key column in `table_A`.
    fk_B : str
        Foreign key column in `table_B`.
    verbose : bool, optional
        Whether to print the statistics (default is False).
    """
    ab, _ = count_of_objets_A_by_objet_B(table_A, pk_A, fk_B)
    ba, _ = count_of_objets_A_by_objet_B(table_A, fk_B, pk_A)

    name_A = table_A.columns.name
    name_B = table_B.columns.name

    _, _, pk_A_not_B, pk_B_not_A = out_of_intersection(table_A, table_B, fk_B)

    ab_min = 0 if len(pk_B_not_A) > 0 else ab.index.min()
    ab_max = ab.index.max()
    ba_min = 0 if len(pk_A_not_B) > 0 else ba.index.min()
    ba_max = ba.index.max()

    print(
        "relation arities : "
        f"[{name_A}]({ab_min}..{ab_max})"
        f"--({ba_min}..{ba_max})[{name_B}]"
    )

    ab = ab.T
    ab.insert(0, "sum", ab.T.sum())

    ba = ba.T
    ba.insert(0, "sum", ba.T.sum())

    if verbose:
        display(ab)
        display(ba)

    return ab, ba, ab_min, ab_max, ba_min, ba_max


""" P7 Revision
"""


def out_of_intersection(
    table_A: pd.DataFrame,
    table_B: pd.DataFrame,
    pk_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the primary keys that are in one table but not the other.

    Parameters
    ----------
    table_A : pd.DataFrame
        First table.
    table_B : pd.DataFrame
        Second table.
    pk_name : str
        Name of the primary key column.

    Returns
    -------
    pk_A : np.ndarray
        Primary keys of `table_A`.
    pk_B : np.ndarray
        Primary keys of `table_B`.
    pk_A_not_B : np.ndarray
        Primary keys of `table_A` that are not in `table_B`.
    pk_B_not_A : np.ndarray
        Primary keys of `table_B` that are not in `table_A`.
    """
    pk_A = table_A[pk_name].drop_duplicates()
    pk_B = table_B[pk_name].drop_duplicates()
    pk_A_not_B = np.setdiff1d(pk_A, pk_B, assume_unique=True)
    pk_B_not_A = np.setdiff1d(pk_B, pk_A, assume_unique=True)
    return pk_A, pk_B, pk_A_not_B, pk_B_not_A


def is_in_A_but_not_in_B_matrix(
    table_dict: Dict[str, pd.DataFrame],
    key: str,
    normalize: bool = False
) -> np.ndarray:
    """
    Calculate the matrix indicating the number of unique values in the
    primary key column of each table that are not present in the primary key
    column of other tables. Returns the matrix as a NumPy array.

    Parameters
    ----------
    table_dict : dict
        A dictionary where the keys are table names (str) and the values are
        the tables as pandas DataFrames.
    key : str
        The name of the primary key column that is common across all the
        tables.
    normalize : bool, optional
        If True, the resulting matrix will contain the relative frequencies 
        of occurrences by normalizing the count of occurrences by the total
        number of unique elements in the union of the two compared columns.
        If False, the count of occurrences will be returned. Default is False.

    Returns
    -------
    numpy.ndarray
        A matrix where the rows and columns correspond to the tables in
        `table_dict`, and each entry indicates the number of unique values in
        the primary key column of the row table that are not present in the
        primary key column of the column table.
    """
    # Construct the unique series of the primary key in each table
    # TODO : a priori, on fait ce qui va être fait par setdiff1d
    # donc il n'est plus certain que ce soit une optimisation, peut être l'opposé
    # timeit!
    pks = [t[key].drop_duplicates() for t in table_dict.values()]

    # Perform the anp.union1d(row_pk, col_pk).shape[0] if normalize else 1
    # TODO changer pour utiliser en une fois is_in_A_and_in_B_matrix si normalize = True
    return np.array([
        [
            (
                np.setdiff1d(row_pk, col_pk, assume_unique=True).shape[0]
                / (np.union1d(row_pk, col_pk).shape[0] if normalize else 1)
            )
            for col_pk in pks
        ]
        for row_pk in pks
    ])


def is_in_A_and_in_B_matrix(
    table_dict: Dict[str, pd.DataFrame],
    key: str
) -> np.ndarray:
    """
    Calculate the matrix indicating the number of unique values in the
    primary key column of each table that are present in the primary key column
    of other tables. Returns the matrix as a NumPy array.

    Parameters
    ----------
    table_dict : dict
        A dictionary where the keys are table names (str) and the values are
        the tables as pandas DataFrames.
    key : str
        The name of the primary key column that is common across all the
        tables.

    Returns
    -------
    numpy.ndarray
        A matrix where the rows and columns correspond to the tables in
        `table_dict`, and each entry indicates the number of unique values in
        the primary key column of the row table that are present in the primary
        key column of the column table.
    """
    # Construct the unique series of the primary key in each table
    # TODO : mm rmq que ci-dessus
    pks = [t[key].drop_duplicates() for t in table_dict.values()]

    # Perform the symmetric union for each couple of tables
    # TODO : optim : fait UP et en déduire DL puis diag
    return np.array([
        [
            np.union1d(row_pk, col_pk).shape[0]
            for col_pk in pks
        ]
        for row_pk in pks
    ])


def display_is_in_A_but_not_in_B_heatmap(
    table_dict: Dict[str, pd.DataFrame],
    key: str,
    normalize: bool = False,
    ratio: float = 1.0
) -> np.ndarray:
    """
    Display the heatmap indicating the number of unique values in the
    primary key column of each table that are not present in the primary
    key column of other tables.

    DEPRECATED Use `display_is_in_A_but_not_in_B_heatmaps` instead

    Parameters
    ----------
    table_dict : dict
        A dictionary where the keys are table names (str) and the values are
        the tables as pandas DataFrames.
    key : str
        The name of the primary key column that is common across all the
        tables.
    normalize : bool, optional
        If True, the resulting matrix will contain the relative frequencies 
        of occurrences by normalizing the count of occurrences by the total
        number of unique elements in the union of the two compared columns.
        If False, the count of occurrences will be returned. Default is False.
    ratio : float, optional
        The aspect ratio of the plot, by default 1.

    Returns
    -------
    np.ndarray
        The matrix of counts (or relative frequencies if `normalize=True`)
        where rows and columns correspond to tables and cells contain the count
        (or relative frequency) of occurrences where a primary key is in one
        table but not in another

    Examples
    --------
    >>> from home_credit.utils import get_table
    >>> from home_credit.load import get_var_descs
    >>> key = "SK_ID_CURR"
    >>> var_descs = get_var_descs()
    >>> table_names = var_descs[var_descs.Column == key].Table
    >>> table_dict = {
    >>>     table_name: get_table(table_name)
    >>>     for table_name in table_names.values
    >>> }
    """
    mx = is_in_A_but_not_in_B_matrix(table_dict, key, normalize)
    labels = table_dict.keys()
    # Draw a heatmap with the numeric values in each cell
    _, ax = plt.subplots(figsize=(8 * ratio, 6 * ratio))
    sns.heatmap(mx,
        xticklabels=labels,  # Pylance False positive
        yticklabels=labels,  # Pylance False positive
        annot=True, fmt=(".2g" if normalize else "n"),
        ax=ax
    )
    title = f"PK (`{key}`) relationship : is in A but not in B"
    set_plot_title(title)
    plt.xticks(rotation=30, ha="right", rotation_mode="anchor")

    # Save and show the plot
    save_and_show(f"{title.lower()}", sub_dir="corr")
    return mx


# Les deux en un
def display_is_in_A_but_not_in_B_heatmaps(
    table_dict: Dict[str, pd.DataFrame],
    key: str,
    single_figsize: Tuple[float, float] = (3, 3),
    ratio: float = 1.25
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Display two heatmaps indicating the number and percentage of unique values
    in the primary key column of each table that are not present in the primary
    key column of other tables.

    Parameters
    ----------
    table_dict : dict
        A dictionary where the keys are table names (str) and the values are
        the tables as pandas DataFrames.
    key : str
        The name of the primary key column that is common across all the
        tables.
    single_figsize : tuple of 2 float, optional
        The size of each individual subplot, by default (3, 3).
    ratio : float, optional
        The aspect ratio of the figure, by default 1.25.

    Returns
    -------
    Tuple of 2 np.ndarray
        Two matrices of counts and relative frequencies where rows and columns
        correspond to tables and cells contain the count or relative frequency
        of occurrences where a primary key is in one table but not in another.

    Examples
    --------
    >>> from pepper.utils import get_table
    >>> from pepper.load import get_var_descs
    >>> key = "SK_ID_CURR"
    >>> var_descs = get_var_descs()
    >>> table_names = var_descs[var_descs.Column == key].Table
    >>> table_dict = {
    >>>     table_name: get_table(table_name)
    >>>     for table_name in table_names.values
    >>> }
    >>> display_is_in_A_but_not_in_B_heatmaps(table_dict, key)
    """
    # Set the title and subtitles
    title = f"PK (`{key}`) relationship :\n“is in $A$ but not in $B$”"
    subtitles = ["Counts", "Percents"]

    # Define the ticklabels
    ticklabels = {
        "labels": table_dict.keys(),
        "rotation": 30, "ha": "right", "rotation_mode": "anchor"
    }

    # Compute the matrices
    mx = is_in_A_but_not_in_B_matrix(table_dict, key)
    umx = is_in_A_and_in_B_matrix(table_dict, key)

    # Create the figure and subplots
    nrows, ncols = 1, 2
    w, h = single_figsize
    figsize = w*ncols*ratio, h*nrows*ratio
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.ravel(axes)

    # Plot the heatmaps
    heatmap_kws = {"annot": True, "annot_kws": {"size": 8}}
    sns.heatmap(mx, fmt="n", ax=axes[0], **heatmap_kws)
    sns.heatmap(100*mx/umx, fmt=".1f", ax=axes[1], **heatmap_kws)

    # Complete the annotations
    [t.set_text(t.get_text() + " %") for t in axes[1].texts]

    # Set the ticklabels and subtitles
    [ax.set_xticklabels(**ticklabels) for ax in axes]
    [ax.set_yticklabels(**ticklabels) for ax in axes]
    [ax.set_title(stl) for ax, stl in zip(axes, subtitles)]
    plt.suptitle(title, fontsize=15)

    # Adjust the spacing between the subplots and save/show the figure
    fig.tight_layout()
    save_and_show(f"{title.lower()}", sub_dir="corr")
    
    # Return the matrices
    return mx, umx


def test_1_is_in_A_but_not_in_B(table_dict, n_iter = 1_000):
    t_a = table_dict["application"]
    pk_A = t_a.SK_ID_CURR

    # Identité
    u_A_1 = pk_A.unique()  # ndarray
    u_A_2 = pk_A.drop_duplicates()  # series
    print("id:", (u_A_1 == u_A_2).all())

    # Rapidité
    t1 = timeit.timeit(lambda: pk_A.unique(), number=n_iter)
    t2 = timeit.timeit(lambda: pk_A.drop_duplicates(), number=n_iter)
    print(f"unique: {t1:.6f} s")
    print(f"drop_duplicates: {t2:.6f} s")

    """
    unique: 8.776817 s
    drop_duplicates: 7.485717 s
    """


def test_2_is_in_A_but_not_in_B(table_dict, n_iter = 1_000):
    t_a = table_dict["application"]
    t_b = table_dict["bureau"]

    pk_A = t_a.SK_ID_CURR
    pk_B = t_b.SK_ID_CURR

    u_A = pk_A.drop_duplicates()
    u_B = pk_B.drop_duplicates()

    fs = [
        lambda: np.array(list(set(u_A) - set(u_B))),
        lambda: np.setdiff1d(u_A, u_B),   # tente d'ordonner par défaut
        lambda: np.setdiff1d(u_A, u_B, assume_unique=True),   # tente d'ordonner si assume_unique=False
        lambda: pd.Index(u_A).difference(u_B), # tente d'ordoner par défaut
        lambda: pd.Index(u_A).difference(u_B, sort=False) # tente d'ordoner par défaut
    ]

    diffs = []
    for f in fs:
        d = f()
        diffs.append(d)
        print(len(d))

    for d in diffs[:3]:
        d.sort()

    for i in range(3, 5):
        diffs[i] = diffs[i].sort_values()

    for d in diffs:
        display(d)

    # Ident
    d_ref = diffs[0]
    for i in range(1, 5):
        print(f"id1{i+1}:", (d_ref == diffs[i]).all())

    # Perf
    for i in range(5):
        print(f"option {i+1}: {timeit.timeit(fs[i], number=n_iter)} s")

    """
    option 1: 125.043920 s
    option 2: 38.728508 s
    option 3: 8.136995 s          <= the best!
    option 4: 34.197448 s
    option 5: 28.507896 s
    """
