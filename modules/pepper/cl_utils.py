""" Clustering utils
"""

from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np


def match_class(y_clu, y_cla):
    """Finds the best matching between true and predicted classes based on
    the size of the intersection between the indices of `y_clu` and `y_cla`.

    Parameters
    ----------
    y_clu : np.ndarray)
        Predicted classes
    y_cla : np.ndarray
        True classes

    Returns
    -------
    class_mapping : np.ndarray
        Best match between the predicted and true classes based on the size of
        the intersection between the indices of `y_clu` and `y_cla`.

    Example
    -------
    >>> mapping = match_class(y_pred, y)  # linear assignement
    >>> y_pred = np.array([mapping[clu] for clu in y_pred])

    See also
    --------
    https://en.wikipedia.org/wiki/Assignment_problem
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    """
    if isinstance(y_clu, list):
        y_clu = np.array(y_clu)
    if isinstance(y_cla, list):
        y_cla = np.array(y_cla)

    n_clusters = np.unique(y_clu).shape[0]
    n_classes = np.unique(y_cla).shape[0]
    match_matrix = np.zeros((n_clusters, n_classes))

    # print(match_matrix)
    for clu in range(n_clusters):
        for cla in range(n_classes):
            intersection = np.intersect1d(
                np.where(y_clu == clu),
                np.where(y_cla == cla)
            )
            match_matrix[clu, cla] = intersection.shape[0]

    # display(match_matrix)
    return linear_assignment(-match_matrix)[1]
