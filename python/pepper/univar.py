## Analyse générique (données quali et quanti)
#  `series_infos` analyse une Series et `dataframe_infos` toutes les Series d'un DataFrame.

from typing import Union, Callable, Tuple
import os
import pandas as pd
import numpy as np

import scipy.stats as sps
from statsmodels import robust
from sklearn.metrics import matthews_corrcoef

import matplotlib.pyplot as plt
import seaborn as sns

#from pepper.data_dict import group, subgroup, domain
from pepper.env import get_tmp_dir
# from pepper_commons import *
from pepper.utils import set_plot_title, save_and_show
from IPython.display import display

def series_infos(s, idx):
    precision = 3
    is_numeric = pd.api.types.is_numeric_dtype(s.dtype)
    is_bool = pd.api.types.is_bool_dtype(s.dtype)
    # is_id : se détecte avec un n_u / n proche de 1, même si numérique
    counts = s.value_counts()
    freqs = s.value_counts(normalize=True)
    n = s.size
    n_na = s.isna().sum()
    n_notna = s.notna().sum()
    n_unique = s.nunique()
    return {
        # Identification, groupe et type
        'idx': idx,
        'group': '<NYI>',  # group(idx),
        'subgroup': '<NYI>',  # subgroup(idx),
        'name': s.name,
        'domain': '<NYI>',  # domain(idx),
        'format': '<NYI>',
        'dtype': s.dtype,
        'astype': '<NYI>',
        'unity': '<NYI>',
        'is_numeric': is_numeric,
        
        # Nombre de valeurs absentes, de valeurs uniques, taux de remplissage
        'n_elts': n,
        'hasnans': s.hasnans,
        'n_unique': n_unique,
        'n_notna': n_notna,
        'n_na': n_na,
        'filling_rate': round(n_notna / n, precision),
        'uniqueness': round(n_unique / n_notna, precision) if n_notna else pd.NA,
        
        # Tendance centrale de la variable numérique
        'val_min': s.min() if is_numeric else None,
        'val_max': s.max() if is_numeric else None,
        'val_mode': str([round(m, 2) for m in s.mode().tolist()[:10]]) if is_numeric else None,
        'val_mean': round(s.mean(), precision) if is_numeric else None,
        'val_trim_mean_10pc': sps.trim_mean(s.dropna().values, 0.1) if is_numeric else None, # TODO test : pas certain que ce ne soit pas s.values
        'val_med': s.median() if is_numeric else None,
        
        # Distribution de la variable numérique
        'val_std': round(s.std(), precision) if is_numeric else None,
        'val_interq_range': (s.quantile(0.75) - s.quantile(0.25)) if is_numeric and not is_bool else None,
        'val_med_abs_dev': robust.scale.mad(s.dropna()) if is_numeric else None,
        'val_skew': round(s.skew(), precision) if is_numeric else None,
        'val_kurt': round(s.kurtosis(), precision) if is_numeric else None,
        'interval': [s.min(), s.max()] if is_numeric else None,
        
        # Distribution de la variable catégorielle
        'modalities': list(counts.index) if not is_numeric else None,
        'mod_counts': list(counts) if not is_numeric else None,
        'mod_freqs': list(freqs) if not is_numeric else None,
        
        # Dimensions, empreinte mémoire, méta-données de l'implémentation
        'shape': s.shape,
        'ndim': s.ndim,
        'empty': s.empty,
        'size': s.size,
        'nbytes': s.nbytes,
        'memory_usage': s.memory_usage,     # non reporté dans le dictionnaire de données
        'flags': s.flags,
        'array container type': type(s.array),
        'values container type': type(s.values)
    }


def dataframe_infos(df):
    infos = [pd.Series(series_infos(df[c], df.columns.get_loc(c))) for c in df.columns]
    return pd.DataFrame(infos)


# Applicable = Callable[..., Union[str, None]]

def data_report(data, csv_filename):
    # build the dataframe version
    data_on_data = dataframe_infos(data)

    # some reductions before export

    # TODO : résoudre le pb Pylance avec le bon cut...: Applicable = lambda ...
    cut_cat_list = lambda x: None if x is None else x[:30] + ['...']
    cut_num_list = lambda x: None if x is None else x[:30] + [sum(x[30:])]

    data_on_data['modalities'] = data_on_data.modalities.apply(cut_cat_list)
    data_on_data['mod_counts'] = data_on_data.mod_counts.apply(cut_num_list)
    data_on_data['mod_freqs'] = data_on_data.mod_freqs.apply(cut_num_list)
    data_on_data.drop(columns=['memory_usage'], inplace=True)

    # save in CSV
    tmp_dir = get_tmp_dir()
    vars_analysis_path = os.path.join(tmp_dir, csv_filename)
    data_on_data.to_csv(vars_analysis_path, sep=',', encoding='utf-8', index=False)

    return data_on_data


def data_report_to_gsheet(data_on_data, spread, sheet_name):
    def null_convert(s):  # représentation du vide.. par du vide
        return s.apply(lambda x: '' if x is None or str(x) in ['nan', '[]', '[nan, nan]'] else x)
    def fr_convert(s):    # gestion de la locale dans un monde numérique encore dominé par les anglo-saxons
        return s.apply(lambda x: str(x).replace(',', ';').replace('.', ','))
    exported = data_on_data.copy()
    exported = exported.apply(null_convert)
    exported['shape'] = exported['shape'].apply(lambda x: '\'' + str(x))  # seuls les initiés savent pourquoi
    exported.loc[:, 'filling_rate':'mod_freqs'] = exported.loc[:, 'filling_rate':'mod_freqs'].apply(fr_convert)
    spread.df_to_sheet(exported, sheet=sheet_name, index=False, headers=False, start='A7')
    # display(exported.loc[:, 'filling_rate':'mod_freqs'])  # un dernier contrôle visuel


""" Cats analysis
"""

def agg_value_counts(
    s: pd.Series,
    agg: Union[None, bool, float, int] = .01,
    dropna: bool = True
) -> pd.DataFrame:
    """Computes the value counts and relative frequencies of a pandas Series.
    
    Parameters
    ----------
    s : pd.Series
        The pandas Series.
    agg : Union[None, bool, float, int], optional
        An optional parameter that specifies how to aggregate the results.
        If None, returns the raw value counts and relative frequencies.
        If a float between 0 and 1, returns a DataFrame with a row containing
        all values whose cumulative proportion of occurrences is less than agg,
        with the remaining values aggregated. If True, determines the threshold
        automatically using the first index at which the proportion is less
        than the sum of the next proportions.
    dropna : bool, default True
        Don't include counts of NaN.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with two columns: 'count', which contains the
        absolute frequency of each value, and 'proportion', which contains the
        relative frequency of each value. If aggregation is used, the last row
        of the DataFrame will show the aggregated counts and proportions.
    
    Examples
    --------
    >>> s = app_amt_req.AMT_REQ_CREDIT_BUREAU_MON.dropna().astype(int)
    >>> display(agg_value_counts(s))
    >>> display(agg_value_counts(s, agg=True))
    >>> display(agg_value_counts(s, agg=.15))
    """
    # Compute the absolute and relative value counts of the series
    abs_vc = s.value_counts(dropna=dropna)
    rel_vc = s.value_counts(normalize=True, dropna=dropna)

    # Combine the absolute and relative value counts into a single DataFrame
    vc = pd.concat([abs_vc, rel_vc], axis=1)
    
    if agg is None or s.dtype == bool:
        # If no aggregation is requested,
        # return the raw value counts and relative frequencies
        return vc
    else:
        # Compute the cumulative sum of the relative freqs in reverse order
        rev_rel_vc = rel_vc.loc[rel_vc.index[::-1]]
        rel_vc_rcumsum = rev_rel_vc.cumsum().loc[rel_vc.index]

        # Determine which values to aggregate
        if isinstance(agg, (int, float)) and not isinstance(agg, bool):
            # If a threshold value is specified, identify the values whose
            # cumulative proportion is less than the threshold
            is_epsilon = rel_vc_rcumsum < agg
        elif isinstance(agg, bool) and agg:
            # If automatic thresholding is requested, identify the first index
            # at which the proportion is less than the sum of the next ones
            """agg_start_idx = np.argmin(rel_vc > rel_vc_rcumsum.shift(-1))
            is_epsilon = vc.index >= agg_start_idx"""
            # Avoid a one bar graph :
            is_sup_to_remainder = rel_vc > rel_vc_rcumsum.shift(-1)
            first_true_idx = np.argmax(is_sup_to_remainder)
            agg_start_idx = first_true_idx + np.argmin(is_sup_to_remainder[first_true_idx:])
            is_epsilon = np.array(
                [False] * agg_start_idx
                + [True] * (vc.shape[0] - agg_start_idx)
            )
            # is_epsilon = vc.index >= agg_start_idx

        # Avoid a one class aggregator
        if is_epsilon.sum() < 2:
            return vc

        # Aggregate the values as necessary
        agg_vc = vc[~is_epsilon].copy()
        eps_vc = vc[is_epsilon].copy()
        eps_vc.sort_index(inplace=True)
        eps_idx = f"{eps_vc.index[0]}:{eps_vc.index[-1]}"
        # don't do this (undesired float cast that must then be fixed by another cast) :
        # agg_vc.loc[eps_idx] = eps_vc.sum()
        # agg_vc.astype({"count": int}, copy=False)
        # the right solution is :
        sums_row = pd.DataFrame(
            {"count": eps_vc["count"].sum(), "proportion": eps_vc["proportion"].sum()},
            index=[eps_idx]
        )
        agg_vc = pd.concat([agg_vc, sums_row])
        agg_vc.index.name = s.name
        return agg_vc


""" P4 : totalement dépassé
def cats_freqs(data, label):
    c = data[label]
    f = c.value_counts()
    f.plot.bar()   # TODO : faire plus sympa à l'aide de SNS
    plt.show()
    return f
"""

# P4 à refaire en mieux
def cats_weighted_freqs(data, label, wlabels, short_wlabels):
    cw = data[[label] + wlabels]
    aggs = {label: 'count'}
    for wlabel in wlabels:
        aggs[wlabel] = 'sum'
    fw = cw.groupby(by=label).agg(aggs)
    fw.columns = ['count'] + short_wlabels
    fw = fw.sort_values(by=fw.columns[1], ascending=False)
    total = fw.sum()
    fw /= total
    fw.plot.bar(figsize=(.5 * fw.shape[0], 5))   # TODO : faire plus sympa à l'aide de SNS
    plt.show()
    return fw



""" Correlation
"""


def target_weights(target: np.ndarray) -> Tuple[float, float]:
    """Calculates the weights of each class in the target array.
    
    Parameters:
    -----------
    target : np.ndarray
        The target variable as a binary array (0s and 1s).
        
    Returns:
    --------
    Tuple[float, float]
        A tuple containing the weights for class 0 and class 1, respectively.
        
    Example:
    --------
    >>> target = np.array([0, 1, 1, 0, 1])
    >>> target_weights(target)
    (1.25, 5.0)
    """
    n = len(target)
    n_1 = (target == 1).sum()
    n_0 = n - n_1
    w_0 = n / (1 + n_0)  # 1 + n to avoid div by 0
    w_1 = n / (1 + n_1)
    return w_0, w_1


def get_sample_weights(target: np.ndarray) -> np.ndarray:
    """Computes the sample weights for each element in the target array.
    
    Parameters:
    -----------
    target : np.ndarray
        The target variable as a binary array (0s and 1s).
        
    Returns:
    --------
    np.ndarray
        An array with the sample weights for each element in the target array.
        
    Example:
    --------
    >>> target = np.array([0, 1, 1, 0, 1])
    get_sample_weights(target)
    array([1.25, 5.  , 5.  , 1.25, 5.  ])
    """
    w_0, w_1 = target_weights(target)
    # Build a ndarray with weight for each sample
    return target * w_1 + (1 - target) * w_0


def wmcc(target: np.ndarray, var: np.ndarray) -> float:
    """Computes the weighted Matthews correlation coefficient between two
    arrays.

    Parameters:
    -----------
    target : np.ndarray
        The target variable as a binary array (0s and 1s).
    var : np.ndarray
        The variable to compare to the target variable, as an array of any type
        with an order relation.

    Returns:
    --------
    float
        The weighted Matthews correlation coefficient between the two arrays.

    Example:
    --------
    >>> target = np.array([0, 0, 1, 1, 1])
    >>> var = np.array([1, 0, 1, 0, 1])
    >>> wmcc(target, var)
    0.25819888974716115
    """
    return matthews_corrcoef(
        target, var,
        sample_weight=get_sample_weights(target)
    )


from scipy import stats
def weighted_kendall_tau(target: np.ndarray, var: np.ndarray) -> float:
    """
    Calculate the weighted Kendall's Tau rank correlation between two arrays.
    
    Parameters:
    -----------
    target : np.ndarray
        The target variable as a binary array (0s and 1s).
    var : np.ndarray
        The variable to compare to the target variable, as an array of any type
        with an order relation.
        
    Returns:
    --------
    float
        The weighted Kendall's Tau rank correlation between the two arrays.
        
    Example:
    --------
    # generate sample data
    >>> target = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    >>> var = np.array([1.2, 2.5, 0.8, 1.5, 1.7, 0.9, 2.1, 1.1, 1.8, 0.7])
    # calculate weighted Kendall's Tau rank correlation
    >>> weighted_kendall_tau(target, var) 
    0.13636363636363635
    """
    w_0, w_1 = target_weights(target)
    corr, p_value = stats.weightedtau(
        target, var,
        weigher=lambda x: w_1 if x else w_0
    )
    print("Weighted Kendall's Tau Rank Correlation : ", corr)
    print("P-value : ", p_value)
    return corr  


def show_correlations(
    data: pd.DataFrame,
    title: str,
    method: str = "pearson",
    ratio: float = 1
) -> pd.DataFrame:
    """Computes pairwise correlation of columns in the input dataframe using
    the specified correlation method and plot the correlation matrix as a
    heatmap.

    Parameters
    ----------
    data : pd.DataFrame
        The input data to compute the correlation matrix on.
    title : str
        The title of the plot.
    method : str, optional
        The correlation method to use, can be one of {'pearson', 'spearman',
        'kendall', 'matthews'}, by default 'pearson'.
    ratio : float, optional
        The aspect ratio of the plot, by default 1.

    Returns
    -------
    pd.DataFrame
        The computed pairwise correlation of columns in the input dataframe.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame({
            "col_1": np.random.normal(0, 1, 100),
            "col_2": np.random.normal(0, 1, 100),
            "target": np.random.binomial(1, 0.5, 100)
        })
    >>> corr_matrix = show_correlations(
    >>>     data, "Correlation matrix", method="spearman"
    >>> )
    """
    data_corr = None
    if method == "matthews":
        data_corr = data.corr(method=wmcc)
    elif method == "kendall":
        data_corr = data.corr(method=weighted_kendall_tau)
    else:
        data_corr = data.corr(method=method)
    
    # Draw a heatmap with the numeric values in each cell
    _, ax = plt.subplots(figsize=(8 * ratio, 6 * ratio))
    sns.heatmap(data_corr, annot=True, ax=ax)
    title = f"{title} ({method})"
    set_plot_title(title)
    plt.ylabel("")
    plt.xlabel("")
    plt.xticks(rotation=30, ha="right", rotation_mode="anchor")

    # Save and show the plot
    save_and_show(f"{title.lower()}", sub_dir="corr")
    return data_corr


def test_wmcc(target):
    display(wmcc(target, target))
    display(wmcc(target, 1 - target))
    m = len(target) // 3
    shuffled_target = target.values.copy()
    np.random.shuffle(shuffled_target[m:2*m])
    display((target == shuffled_target).all())
    display(wmcc(
        target,
        shuffled_target
    ))
