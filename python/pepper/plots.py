from typing import Tuple  # Optional, Union, 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from matplotlib.figure import Figure
from pepper.utils import save_and_show
import locale
locale.setlocale(locale.LC_ALL, '')


def show_cat_mod_counts(
    cat,
    order=None,
    ax=None,
    rotate_labels=True,
    title=None
):
    fig = None
    if ax is None:
        figsize = (max(10, cat.nunique()), 6)
        fig, ax = plt.subplots(figsize=figsize)

    ax = sns.countplot(x=cat, order=order, ax=ax)

    if title is None:
        title = "Jane DOE" if cat.name is None else cat.name
    ax.set_title(title, pad=10)
    ax.set_xlabel("")

    # Rotate the x-axis tick labels and set their alignment.
    if rotate_labels:
        plt.setp(
            ax.get_xticklabels(),
            rotation=30, ha="right", rotation_mode="anchor"
    )
    
    # Add relative frequency values above bars
    total = float(len(cat))  # total number of observations
    for p in ax.patches:
        height = p.get_height()  # height of each bar which is the count
        pct = 100 * height / total  # relative frequency in percentage
        ax.annotate(
            f"{int(height):n} ({pct:.1f} %)", (p.get_x() + p.get_width() / 2., height), 
            ha="center", va="center", fontsize=7, weight="bold", # color="black", 
            xytext=(0, 7), textcoords="offset points"
        )
    
    if fig is not None:
        fig.tight_layout()
        # plt.show()
        # title = "Jane DOE cat" if cat.name is None else cat.name
        save_and_show(f"cat_modalities_counts_{title.lower()}", sub_dir="dataxplor")
    

def show_cat_mod_counts_gallery(
    data,
    columns,
    ncols=2,
    rotate_labels=True
):
    n = len(columns)
    nrows = n // ncols + (n % ncols > 0)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows))
    axes = np.ravel(axes)
    for ax, cat in zip(axes, columns):
        order = data[cat].value_counts().index
        show_cat_mod_counts(data[cat], order=order, ax=ax, rotate_labels=rotate_labels)
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.tight_layout()
    # plt.suptitle(data.columns.name)
    plt.show()


def lin_log_tetra_histplot(
    s: pd.Series,
    color: str = "gold",
    single_figsize: Tuple[int, int] = (4, 3),
    title: str = None,
    max_bins: int = int(8*365.25)
) -> None:
    """Plots a four histograms of the same data with two axes in linear scale
    and two axes in log scale.

    Parameters
    ----------
    s : pd.Series
        A series to plot.
    color : str, optional
        The color of the histograms. Defaults to "gold".
    single_figsize : Tuple[int, int], optional
        The size of a single subplot figure. Defaults to (4, 3).
    title : str, optional
        The title of the figure. Defaults to None.
    max_bins : in, optional
        Max number of bins

    Returns
    -------
    None

    """
    # Determine the number of bins
    bins = min(int(1 + s.max() - s.min()), max_bins)

    # Create the figure and subplots
    nrows = ncols = 2
    w, h = single_figsize
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(w * ncols, h * nrows)
    )

    # Plot the histograms
    [s.plot.hist(bins=bins, color=color, ax=ax) for ax in np.ravel(axes)]

    # Add gridlines to each subplot
    [ax.grid(linewidth=0.2) for ax in np.ravel(axes)]

    # Set the x- and y-axis scales for the appropriate subplots
    [ax.set_xscale("log") for ax in axes[:, 1]]
    [ax.set_yscale("log") for ax in axes[1, :]]

    # Format the y-axis tick labels for the top row of subplots
    [ax.ticklabel_format(
        axis="y", style="sci", useMathText=True, scilimits=(-2, 2)
    ) for ax in axes[0, :]]

    # Add titles to each subplot
    nm = lambda k: "log" if k else "lin"
    [[axes[i, j].set_title(f"{nm(j)}-{nm(i)}") for i in [0, 1]] for j in [0, 1]]

    # Add a title to the figure
    if title is None:
        title = "Jane DOE series" if s.name is None else s.name
    plt.suptitle(title, fontsize=15, weight="bold")

    # Adjust the spacing between the subplots and save/show the figure
    fig.tight_layout()
    save_and_show(f"linlog_tetra_hist_{title.lower()}", sub_dir="dataxplor")

