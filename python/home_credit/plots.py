from typing import Optional, Dict, Tuple, Callable

import locale
locale.setlocale(locale.LC_ALL, "")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
# from matplotlib.figure import Figure

from pepper.utils import save_and_show

from home_credit.utils import get_class_label_name_map


def by_target_class_kdeplot(
    targeted_data: pd.DataFrame,
    var_name: str,
    transf_var: Optional[Callable[[pd.Series], pd.Series]] = None,
    var_rename: Optional[str] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (8, 3)
) -> None:
    """
    Plot the kernel density estimation (KDE) distribution of a variable by
    target class.

    Parameters
    ----------
    targeted_data : pd.DataFrame
        The dataset to plot.
    var_name : str
        The name of the variable to plot.
    transf_var : callable, optional
        A function to apply to `var_name` before plotting.
        Default to None.
    var_rename : str, optional
        The label to use for `var_name` in the plot.
        Default to None.
    ax : Axes, optional
        The Matplotlib Axes object to use for the plot.
        If not provided, a new figure and Axes will be created.
        Default is None.
    figsize : Tuple[int, int], optional
        The size of the figure if `ax` is not provided.
        Default is (8, 3).

    Example
    -------
    >>> by_target_class_kdeplot(
    >>>     data, "DAYS_BIRTH",
    >>>     transf_var=lambda s: -s / 365.25,
    >>>     var_rename="Age (years)",
    >>>     ax=None,
    >>>     figsize=(8, 3)
    >>> )
    """
    target = targeted_data.TARGET
    var = (
        transf_var(targeted_data[var_name])
        if transf_var is not None
        else targeted_data[var_name]
    )

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # plt.figure(figsize=(8, 3))

    # KDE plots of loans depending on each status (target)
    colors = {-1: "yellow", 0: "green", 1: "red"}
    for cl, name in get_class_label_name_map().items():
        sns.kdeplot(var[target == cl], label=name, color=colors[cl], ax=ax)

    # Add gridline to plot
    ax.grid(linewidth=0.2)

    # Labeling of plot
    vname = var_name if var_rename is None else var_rename
    ax.set_xlabel(vname)
    ax.set_ylabel("Density")
    ax.legend()
    
    # Adjust the spacing between the subplots and save/show the figure
    if fig is not None:
        title = f"KDE distribution of {vname} by target"
        ax.set_title(title)
        plt.tight_layout()
        save_and_show(f"{title.lower()}", sub_dir="dataxplor_by_target")


def datablock_by_target_class_kdeplot(
    targeted_data_block: pd.DataFrame,
    var_renames: Optional[Dict[str, str]] = None,
    transf_var: Optional[callable] = None,
    single_figsize: Tuple[int, int] = (8, 3),
    title: Optional[str] = None
) -> None:
    """
    Plot the kernel density estimation (KDE) distribution of multiple variables
    by target class for a data block.

    This function generates multiple KDE plots, each showing the distribution
    of a variable by target class. It is especially useful for exploring the
    relationship between multiple features and the target variable in a
    structured dataset.

    Parameters
    ----------
    targeted_data_block : pd.DataFrame
        The dataset block containing the variables to plot and the target variable.
    var_renames : dict, optional
        A dictionary mapping variable names to custom labels for the plots.
        Default is None, which uses the variable names as labels.
    transf_var : callable, optional
        A function to apply to each variable before plotting.
        Default is None.
    single_figsize : Tuple[int, int], optional
        The size of each individual subplot figure.
        Default is (8, 3).
    title : str, optional
        The title for the overall figure.
        Default is None.

    Example
    -------
    >>> datablock_by_target_class_kdeplot(
    >>>     data_block, var_renames={"DAYS_BIRTH": "Age"},
    >>>     transf_var=lambda s: -s / 365.25,
    >>>     single_figsize=(8, 3),
    >>>     title="KDE distribution of variables by target"
    >>> )
    """
    var_names = list(targeted_data_block.columns)
    var_names.remove("TARGET")
    if var_renames is None:
        var_renames = {}
    # Create the figure and subplots
    ncols = 1
    nrows = len(var_names)
    w, h = single_figsize
    _, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(w * ncols, h * nrows)
    )
    axes = np.ravel(axes)
    # var_renames = var_renames if var_renames is not None else var_names
    for ax, var_name in zip(axes, var_names):
        by_target_class_kdeplot(
            targeted_data_block, var_name,
            transf_var=transf_var,
            var_rename=var_renames.get(var_name),
            ax=ax, figsize=single_figsize
        )

    # Add a title to the figure
    if title is None:
        title = "KDE distribution by target value\n"
    plt.suptitle(title, fontsize=15, weight="bold")

    # Adjust the spacing between the subplots and save/show the figure
    plt.tight_layout()
    save_and_show(f"{title.lower()}", sub_dir="dataxplor")
