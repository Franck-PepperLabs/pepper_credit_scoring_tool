from typing import (
    Optional, Union, Any, Callable, List, Tuple, Dict
)

import os, time
from sys import getsizeof
from datetime import datetime, timedelta

import locale
import calendar

from itertools import zip_longest
import re

from IPython.display import display, clear_output
from IPython.core.display import Markdown, HTML

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
# Set the style of plots
plt.style.use('dark_background')
# plt.style.use('fivethirtyeight')
# plt.style.use('ggplot')


from pepper.env import get_img_dir


""" Locale
"""

def get_default_locale() -> str:
    """
    Get the default locale string for formatting.

    Returns
    -------
    str
        The default locale string, e.g., "fr_FR.UTF-8".
    """
    return "fr_FR.UTF-8"


# Ex. f"{123456789:n}" : 123 456 789
locale.setlocale(locale.LC_ALL, get_default_locale())


def get_weekdays(target_locale: str = None) -> list:
    """
    Get the weekdays' names in the specified or default locale.

    Parameters
    ----------
    target_locale : str, optional
        The target locale string for formatting the weekdays, e.g., "en_US.UTF-8".
        If not provided, the default locale will be used.

    Returns
    -------
    list of str
        A list of weekday names in uppercase for the specified or default locale.

    Example
    -------
    >>> get_weekdays("en_US.UTF-8")
    ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
    """
    if target_locale is None:
        target_locale = "en_US.UTF-8"
    locale.setlocale(locale.LC_ALL, target_locale)
    weekdays = [day.upper() for day in calendar.day_name]
    locale.setlocale(locale.LC_ALL, get_default_locale())
    return weekdays


def cls():
    """Clears the output of the current cell receiving output."""
    clear_output(wait=True)


"""Simple functions for pretty printing text.

Functions
---------
bold(s):
    Returns a string `s` wrapped in ANSI escape codes for bold text.
italic(s):
    Returns a string `s` wrapped in ANSI escape codes for italic text.
cyan(s):
    Returns a string `s` wrapped in ANSI escape codes for cyan text.
magenta(s):
    Returns a string `s` wrapped in ANSI escape codes for magenta text.
red(s):
    Returns a string `s` wrapped in ANSI escape codes for red text.
green(s):
    Returns a string `s` wrapped in ANSI escape codes for green text.
print_title(txt):
    Prints a magenta title with the text `txt` in bold font.
print_subtitle(txt):
    Prints a cyan subtitle with the text `txt` in bold font.
"""


def bold(s: object) -> str:
    return "\033[1m" + str(s) + "\033[0m"


def italic(s: object) -> str:
    return "\033[3m" + str(s) + "\033[0m"


def cyan(s: object) -> str:
    return "\033[36m" + str(s) + "\033[0m"


def magenta(s: object) -> str:
    return "\033[35m" + str(s) + "\033[0m"


def red(s: object) -> str:
    return "\033[31m" + str(s) + "\033[0m"


def green(s: object) -> str:
    return "\033[32m" + str(s) + "\033[0m"


def print_title(txt: str) -> None:
    print(bold(magenta('\n' + txt.upper())))


def print_subtitle(txt: str) -> None:
    print(bold(cyan('\n' + txt)))


def print_subsubtitle(txt: str) -> None:
    print(italic(green('\n' + txt)))


def display_key_val(key, val):
    print(f"{bold(key)}: ", end="")
    # TODO : complete with other type cases
    if isinstance(val, int):
        print(f"{val:n}")
    else:
        print(val)


def display_dataframe_in_markdown(
    data: pd.DataFrame,
    show_index: bool = False
) -> None:
    """
    Display a DataFrame as a Markdown table.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to display as a table in Markdown format.
    show_index : bool, optional (default=False)
        If True, display the index/indices of the DataFrame as a column(s).

    Returns
    -------
        None.
    """
    strize = lambda s: [str(x).replace("|", "\|") for x in s]
    pipeize = lambda s: ("|" + "|".join(s) + "|").replace("None|", "_|")

    data_cols = strize(data.columns.tolist())

    # Creating the table header row
    index_cols = strize(data.index.names) if show_index else []
    header_cols = index_cols + data_cols
    header = pipeize(header_cols)
    separator = pipeize(["-"] * len(header_cols))

    # Creating the subsequent rows
    rows = [
        pipeize(strize(row[0] + row[1:] if show_index else row))
        for row in data.itertuples(index=show_index)
    ]

    # Combining the header, separator and rows to create the complete table
    table = "\n".join([header, separator] + rows)

    # Displaying the table in Markdown format
    display(Markdown(table))


""" Plot
"""

def clean_filename(filename: str) -> str:
    """
    Remove all characters that are not allowed in a filename from the input string.

    Parameters
    ----------
    filename : str
        The string to be cleaned.

    Returns
    -------
    str
        The cleaned string.
    """
    return (
        re.sub(r"[\r\n]+|[^\w\s-]", "", filename)
        .strip().replace(" ", "_").lower()
    )


# Styled title
def set_plot_title(title):
    plt.title(title, fontsize=15, weight="bold", pad=15)


def display_file_link(
    filepath: str,
    desc: Optional[str] = None
) -> None:
    """
    Display a clickable file link to the given file path.

    Parameters
    ----------
    filepath : str
        The file path to the file being linked to.
    desc : Optional[str], default None
        An optional description to add before the link.

    Returns
    -------
    None
        This function does not return anything, but displays the link in the
        notebook.

    Examples
    --------
    >>> display_file_link("path/to/my_file.txt")
    Displays a clickable link to the file "my_file.txt".

    >>> display_file_link("path/to/my_file.txt", "Link to my file")
    Displays the text "Link to my file:"
    followed by a clickable link to the file "my_file.txt".
    """
    filename = os.path.basename(filepath)
    html = \
    r"""<style>.nb-Link {{
        font-family: var(--jp-ui-font-family)!important;
        font-size: var(--jp-ui-font-size1)!important;
    }}</style>"""
    if desc is not None:
        html += f"{desc}"
    html += f"<a href='{filepath}')>{filename}</a>"
    display(HTML(html))


def save_and_show(file_name, sub_dir=None, file_ext="png", timestamp=True, return_filepath=False):
    file_name = clean_filename(file_name)
    root_dir = get_img_dir() + "/"   # "../img/"
    sub_dir = sub_dir or ""
    if len(sub_dir) > 0 and sub_dir[-1] != "/":
        sub_dir += "/"
    path = root_dir + sub_dir
    create_if_not_exist(path)
    file_ext = file_ext or "png"
    if len(file_ext) > 0 and file_ext[0] != ".":
        file_ext = "." + file_ext
    if timestamp:
        file_name += datetime.now().strftime("_%Y_%m_%d_%H_%M_%S_%f")
    plt.savefig(
        f'{path}{file_name}{file_ext}',
        #facecolor='white',
        bbox_inches='tight',
        dpi=300   # x 2
    )
    plt.show()
    # print(f"save_and_show_savefig({dir}{file_name}{file_ext})")
    #file = f"{file_name}{file_ext}"
    full_path = f"{path}{file_name}{file_ext}"
    display_file_link(full_path, "<b>Figure</b> saved 🔗 ")
    if return_filepath:
        return full_path



""" Dataset discovery
"""


def discrete_stats(
    data: pd.DataFrame,
    name: Optional[Union[str, None]] = None
) -> pd.DataFrame:
    """
    Calculate discrete statistics of a pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data to compute statistics for.
    name : str, optional
        A string indicating the name of the dataset, by default None.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the following statistics for each column of the
        input DataFrame: [count, unique_count, na_count, filling_rate,
        variety_rate] as [n, n_u, n_na, fr, vr].

    Raises
    ------
    TypeError
        If data is not a pandas DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from pepper_utils import discrete_stats
    >>> data = pd.DataFrame({'A': [1, 2, 3, np.nan], 'B': ['cat', 'dog', 'dog', 'dog']})
    >>> discrete_stats(data, name='test')
             n  n_u  n_na     Fill rate   Diversity rate   dtypes
    test
    A        3    3     1      0.750000         0.750000  float64
    B        3    2     0      1.000000         0.666667   object
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data should be a pandas DataFrame.")

    n = data.count()
    n_u = data.nunique()
    n_na = data.isna().sum()
    stats = pd.DataFrame({
        "n": n,
        "n_u": n_u,
        "n_na": n_na,
        "Fill rate": n / (n + n_na),
        "Diversity rate": n_u / n,
        "dtypes": data.dtypes
    }, index=list(data.columns))

    stats.index.name = name if name is not None else data.index.name
    return stats



def plot_discrete_stats(
    stats: pd.DataFrame,
    precision: float = .1,
    ratio: float = 1.0
) -> None:
    """Plots a stacked bar chart and a scatter plot of the input data.

    Parameters
    ----------
    stats : pandas.DataFrame
        A DataFrame containing the statistics to be plotted.
    precision : float, optional
        A number specifying the smallest possible value for Shannon entropy,
        by default 0.1.
    ratio : float, optional
        The ratio of the figure width to height. Defaults to 1.0.

    Raises
    ------
    TypeError
        If stats is not a pandas DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from pepper_utils import plot_discrete_stats, discrete_stats
    >>> data = pd.DataFrame({'A': [1, 2, 3, np.nan], 'B': ['cat', 'dog', 'dog', 'dog']})
    >>> stats = discrete_stats(data, name='test')
    >>> plot_discrete_stats(stats)
    """
    if not isinstance(stats, pd.DataFrame):
        raise TypeError("Input data should be a pandas DataFrame.")
    table_name = stats.index.name
    if table_name is None:
        table_name = "Jane DOE"
    filling_rate = stats[["Fill rate", ]].copy()
    na_rate = 1 - filling_rate["Fill rate"]
    filling_rate.insert(0, "NA_", na_rate)
    filling_rate = filling_rate * 100
    filling_rate.columns = ["NA", "Filled"]

    shannon_entropy = stats["Diversity rate"]
    shannon_entropy = np.maximum(shannon_entropy * 100, precision)

    # Create stacked bar chart
    _, ax1 = plt.subplots(figsize=(ratio * 8, 4))
    filling_rate.plot(kind="bar", stacked=True, color=["lightcoral", "lightgreen"], ax=ax1)

    #ax1 = filling_rate.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'])
    legend1 = ax1.legend(["NA", "Filled"], loc="upper left", bbox_to_anchor=(1, 1))
    plt.gca().add_artist(legend1)

    # Add scatter plot for Shannon entropy
    ax2 = plt.scatter(
        np.arange(len(filling_rate)),  # x-coordinates
        shannon_entropy,               # y-coordinates
        s=200,                         # size of the points
        color="black",
        edgecolors="white"
    )
    plt.legend([ax2], ["Diversity rate"], loc="upper left", bbox_to_anchor=(1, .8))

    plt.yscale("log")
    plt.ylim(precision, 100)

    # Axis titles
    plt.ylabel("Fill & Diversity rates")
    plt.xlabel("")

    # Rotate x-axis labels
    plt.xticks(rotation=30, ha="right")

    # Add overall title
    # plt.title(f'Discrete statistics of `{table_name}` table', fontsize=15, weight="bold", pad=15)
    set_plot_title(f"Discrete statistics of `{table_name}` table")

    # Save and show the plot
    save_and_show(f"discrete_stats_{table_name.lower()}", sub_dir="discrete_stats")


def show_discrete_stats(
    data: pd.DataFrame,
    name: Optional[str] = None,
    precision: float = .1,
    ratio: float = 1.0
) -> None:
    """Shows the discrete statistics of a Pandas DataFrame and plot them.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to analyze.
    name : str, optional
        A name to give to the index of the returned DataFrame, by default None.
    precision : float, optional
        A number specifying the smallest possible value for Shannon entropy,
        by default 0.1.
    ratio : float, optional
        The ratio of the figure width to height. Defaults to 1.0.

    Returns
    -------
    None
    """
    if name is None:
        name = data.columns.name
    stats = discrete_stats(data, name=name)
    display(stats)
    plot_discrete_stats(stats, precision, ratio)


""" Generics
"""


def create_if_not_exist(dir: str) -> None:
    """
    Create a directory if it doesn't already exist.

    Parameters
    ----------
    dir : str
        The directory path to create.

    Returns
    -------
    None
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def for_all(
    f: Callable[..., Any],
    args_vect: Optional[List[Optional[Tuple[Any, ...]]]] = None,
    kwargs_vect: Optional[List[Optional[Dict[str, Any]]]] = None,
    const_args: Optional[Tuple[Any, ...]] = None,
    const_kwargs: Optional[Dict[str, Any]] = None
) -> Union[None, Any, List[Any]]:
    """
    Apply a function to a vector of arguments and/or keyword arguments.

    If `args_vect` and `kwargs_vect` are provided, apply
    `f(*args_vect[i], **kwargs_vect[i])` to each corresponding pair
    `(args_vect[i], kwargs_vect[i])`. If only `args_vect` or `kwargs_vect`
    is provided, `f(*args_vect[i])` or `f(**kwargs_vect[i])` is applied to each
    element.

    If `const_args` or `const_kwargs` are provided, their values are used as
    additional arguments and/or keyword arguments in each function call.

    Parameters
    ----------
    f : callable
        The function to apply to each set of arguments.
    args_vect : list of tuple or None, default=None
        A vector of positional arguments. Each element of the list is a tuple
        of arguments to be passed to the function. If `None`, `kwargs_vect`
        must be provided.
    kwargs_vect : list of dict or None, default=None
        A vector of keyword arguments. Each element of the list is a dictionary
        of keyword arguments to be passed to the function. If `None`,
        `args_vect` must be provided.
    const_args : tuple or None, default=None
        Additional positional arguments to pass to the function in each call.
        If `None`, no additional positional arguments are used.
    const_kwargs : dict or None, default=None
        Additional keyword arguments to pass to the function in each call.
        If `None`, no additional keyword arguments are used.

    Returns
    -------
    None or any or list of any
        If `args_vect` and `kwargs_vect` are `None` and `const_args` and
        `const_kwargs` are `None`, `None` is returned. Otherwise, a list of
        return values from `f` is returned.

        If `f` returns a tuple, the list of return values is transposed such
        that the i-th element of the j-th tuple returned by `f` is the j-th
        element of the i-th tuple in the returned list.

    Examples
    --------
        >>> for_all(len, args_vect=[('abc',), ('def', 'ghi'), ()])
        [1, 2, 0]
        >>> for_all(len, kwargs_vect=[{'x': 'abc'}, {'x': 'def', 'y': 'ghi'}, {}])
        [3, 3, 0]
        >>> for_all(lambda x, y: x + y, args_vect=[(1, 2), (3, 4)], const_args=(10,))
        [13, 14]
        >>> for_all(lambda x, y=0: x + y, args_vect=[(1,), (2,)], kwargs_vect=[{'y': 10}, {}])
        [11, 2]
        >>> for_all(lambda x, y: (x, y), args_vect=[(1, 2), (3, 4), (5,)], const_kwargs={'y': 10})
        [(1, 12), (3, 14), (..
    """
    # Handle the case where no arguments are given
    if (
        args_vect is None
        and kwargs_vect is None
        and const_args is None
        and const_kwargs is None
    ):
        return None

    # Ensure that constant arguments are a tuple
    if not (const_args is None or isinstance(const_args, tuple)):
        const_args = (const_args,)

    # Handle the case where only constant arguments are given
    if args_vect is None and kwargs_vect is None:
        if const_args is None and const_kwargs is not None:
            return f(**const_kwargs)
        if const_kwargs is None and const_args is not None:
            return f(*const_args)

    # Apply the function to all combinations of variable and constant arguments
    def call_f(args, kwargs):
        new_args = ()
        if args is not None:
            if not isinstance(args, tuple):
                args = (args,)
            new_args += args
        if const_args is not None:
            new_args += const_args
        new_kwargs = {}
        if const_kwargs is not None:
            new_kwargs.update(const_kwargs)
        if kwargs is not None:
            new_kwargs.update(kwargs)
        return f(*new_args, **new_kwargs)

    results = None
    if args_vect is None and kwargs_vect is not None:
        results = [call_f(None, kwargs) for kwargs in kwargs_vect]
    elif kwargs_vect is None and args_vect is not None:
        results = [call_f(args, None) for args in args_vect]
    elif args_vect is not None and kwargs_vect is not None:
        results = [
            call_f(args, kwargs)
            for args, kwargs in zip(args_vect, kwargs_vect)
        ]

    # If the function returns a tuple, we zip the output
    if (
        results is not None
        and len(results) > 0
        and isinstance(results[0], tuple)
    ):
        results = list(zip_longest(*results))

    # Returns nothing if f is clearly a procedure (never returns anything)
    if results is not None and any(result is not None for result in results):
        return results



""" Memory assets
"""


def format_iB(n_bytes: int) -> Tuple[float, str]:
    """Returns a tuple with the amount of memory and its unit in bytes.

    Parameters
    ----------
    n_bytes : int
        The number of bytes to format.

    Returns
    -------
    Tuple (float, str)
        A tuple containing the amount of memory and its unit in bytes.
    """
    KiB = 2**10
    MiB = KiB * KiB
    GiB = MiB * KiB
    TiB = GiB * KiB
    if n_bytes < KiB:
        return n_bytes, 'iB'
    elif n_bytes < MiB:
        return round(n_bytes / KiB, 3), 'KiB'
    elif n_bytes < GiB:
        return round(n_bytes / MiB, 3), 'MiB'
    elif n_bytes < TiB:
        return round(n_bytes / GiB, 3), 'GiB'
    else:
        return round(n_bytes / TiB), 'TiB'


def print_memory_usage(data: pd.DataFrame) -> None:
    """Prints the memory usage of a pandas DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The pandas DataFrame for which to print the memory usage.
    """
    mem_usage = data.memory_usage()
    print("mem_usage :", *format_iB(mem_usage.sum()))
    # print("getsizeof :", *format_iB(getsizeof(data)))
    print("__sizeof__:", *format_iB(data.__sizeof__()))


def get_file_size(file_path: str) -> int:
    """Returns the size of a file in bytes.

    Parameters
    ----------
    file_path : str
        The path to the file.

    Returns
    -------
    int
        The size of the file in bytes.
    """
    return os.path.getsize(file_path)


def print_file_size(file_path: str) -> None:
    """
    Print the size of a file in bytes.

    Parameters
    ----------
    file_path : str
        The path to the file.
    """
    file_size = get_file_size(file_path)
    print("file size:", *format_iB(file_size))


""" Performances
"""


"""def get_start_time() -> float:
    r""DEPRECATED, use `time.time` instead

    Return the current time.

    Returns
    -------
    float
        The current time.
    ""
    return time.time()"""


def pretty_timedelta_str(dt: float, n_significant: int = 3) -> str:
    """Returns a pretty formatted string of a timedelta.

    Parameters
    ----------
    dt : float
        The timedelta to format.
    n_significant : int, default=3
        The number of significant units to include in the formatted string.

    Returns
    -------
    str
        A pretty formatted string of the timedelta.

    Example
    -------
    >>> import time
    >>> from datetime import timedelta
    >>> pretty_timedelta_str(timedelta(seconds=123456.789))
    '1d, 10h, 17m, 36s, 789ms'
    """
    td = timedelta(seconds=dt)
    d = td.days
    h = td.seconds // 3600
    m = (td.seconds // 60) % 60
    s = td.seconds % 60
    ms, mus = divmod(td.microseconds, 1000)
    vals = [d, h, m, s, ms, mus]
    units = ['d', 'h', 'm', 's', 'ms', 'mus']
    first = next(i for i, j in enumerate(vals) if j)
    vals = vals[first:first+n_significant]
    units = units[first:first+n_significant]
    units_str = [f"{v} {u}" for v, u in zip(vals, units) if v > 0]
    return ", ".join(units_str)


def print_time_perf(
    what: str = '',
    where: str = '',
    time: Optional[float] = None
) -> None:
    """Prints a time perf info.

    Parameters
    ----------
    what : str, optional
        A description of what was done, by default ''
    where : str, optional
        A description of where it was done, by default ''
    time : float, optional
        The time of the operation

    Returns
    -------
        None
    """
    # dt = time.time() - start_time
    if what:
        print(what)
    if where:
        print(where)
    if time:
        print(f"in {pretty_timedelta_str(time)}")
    # return dt

