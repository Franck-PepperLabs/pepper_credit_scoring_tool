import inspect
from IPython.display import display
from pepper.utils import (
    print_title,
    print_subtitle,
    print_subsubtitle,
    display_key_val
)

# Source :
# https://github.com/Franck-PepperLabs/
# pepper_automatic_product_classification_engine/
# blob/main/notebooks/tx_pipeline.py

def tx(verbosity: int, text: str) -> None:
    """Prints a text if the verbosity level is greater than 1.

    Parameters
    ----------
    verbosity : int
        The verbosity level.
    subtitle : str
        The text to print.
    """
    if verbosity is None:
        return
    if verbosity > 1:
        display(text)


def kv(verbosity: int, key: str, val: str) -> None:
    """Prints a text if the verbosity level is greater than 1.

    Parameters
    ----------
    verbosity : int
        The verbosity level.
    subtitle : str
        The text to print.
    """
    if verbosity is None:
        return
    if verbosity > 1:
        display_key_val(key, val)


def tl(verbosity: int, title: str) -> None:
    """Prints the title if the verbosity level is greater than 0.
    If the verbosity level is 1 simple print, else pretty print

    Parameters
    ----------
    verbosity : int
        The verbosity level.
    title : str
        The title to print.
    """
    if verbosity is None or verbosity < 1:
        return
    elif verbosity == 1:
        display(title)
    elif verbosity == 2:
        print_subsubtitle(title)
    elif verbosity == 3:
        print_subtitle(title)
    elif verbosity > 3:
        print_title(title)


def stl(verbosity: int, subtitle: str) -> None:
    """Prints the subtitle if the verbosity level is greater than 1.

    Parameters
    ----------
    verbosity : int
        The verbosity level.
    subtitle : str
        The subtitle to print.
    """
    if verbosity is None or verbosity < 2:
        return
    elif verbosity == 2:
        display(subtitle)
    elif verbosity == 3:
        print_subsubtitle(subtitle)
    elif verbosity > 3:
        print_subtitle(subtitle)


def sstl(verbosity: int, subsubtitle: str) -> None:
    """Prints the sub-subtitle if the verbosity level is greater than 2.

    Parameters
    ----------
    verbosity : int
        The verbosity level.
    subsubtitle : str
        The sub-subtitle to print.
    """
    if verbosity is None or verbosity < 3:
        return
    elif verbosity == 3:
        display(subsubtitle)
    elif verbosity > 3:
        print_subsubtitle(subsubtitle)



def this_f_name() -> str:
    """
    Returns the name of the calling function.

    Returns:
    str: The name of the calling function as a string.

    Example:
    >>> def my_function():
    ...     return this_f_name()
    >>> my_function()
    'my_function'
    """
    return inspect.stack()[1][3]
