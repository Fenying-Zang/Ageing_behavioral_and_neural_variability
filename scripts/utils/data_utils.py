"""
Utility functions for data handling, shuffling, statistics, and age labeling.

Functions
---------
- load_filtered_recordings : Load list of sessions/probes after QC.
- normalize_units          : Min-max normalize neural data across units.
- shuffle_labels_perm      : Permutation shuffling for labels (age, sessions).
- fdr_correct_by_group     : Apply FDR correction within groups.
- bf_gaussian_via_pearson  : Bayes Factor via Pearson correlation.
- interpret_bayes_factor   : Translate BF value into qualitative interpretation.
- add_age_group            : Add categorical age groups (young/old) and scaled age.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os
import pingouin as pg
import math
import config as C


def load_filtered_recordings(datapath=C.DATAPATH, filename = 'BWM_LL_release_afterQC_df.csv'):
    """
    Load a pre-filtered recordings table (after QC).

    Parameters
    ----------
    datapath : Path
        Base path to data folder (default: C.DATAPATH).
    filename : str
        CSV file with filtered sessions/probes.

    Returns
    -------
    recordings_filtered : pd.DataFrame
        Loaded DataFrame.
    """
    try:
        recordings_filtered = pd.read_csv( datapath / f"{filename}" )
    except Exception as err:
        print(f'errored: {err}')
        recordings_filtered =np.nan
    return recordings_filtered


def normalize_units(matrix):
    """
    Normalize neural activity per unit (min-max scaling).

    Parameters
    ----------
    matrix : np.ndarray
        Shape (trials, units, timepoints).

    Returns
    -------
    result : np.ndarray
        Normalized array with same shape, values in [0, 1].
    """
    trials, units, timepoints = matrix.shape
    result = np.zeros_like(matrix)
    
    for i in range(units):
        data = matrix[:, i, :].flatten()
        min_val, max_val = np.min(data), np.max(data)

        # max-min normalization if max_val> min_val
        if max_val > min_val:
            result[:, i, :] = (matrix[:, i, :] - min_val) / (max_val - min_val)

        else:
            result[:, i, :] = np.zeros_like(matrix[:, i, :]) 
    
    return result


def shuffle_labels_perm(labels1, labels2, n_permut=1, shuffling='labels1_based_on_2', n_cores=1, random_state=None):
    """
    Shuffle labels for permutation testing.

    Modes
    -----
    - 'labels1_based_on_2': Shuffle labels1 across groups defined by labels2.
    - 'labels1_global': Shuffle labels1 globally.

    Parameters
    ----------
    labels1 : array-like
        Values to shuffle (e.g., age labels).
    labels2 : array-like or None
        Grouping variable (e.g., session ID).
    n_permut : int
        Number of permutations.
    shuffling : str
        Shuffling mode ('labels1_based_on_2' or 'labels1_global').
    n_cores : int
        Number of CPU cores for parallelism.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    permuted_labels1 : np.ndarray or list of np.ndarray
        Shuffled label arrays.
    labels2 : array-like
        Unchanged labels2.
    """
    labels1 = np.array(labels1)
    if labels2 is not None:
        labels2 = np.array(labels2)

    rng = np.random.default_rng(seed=random_state)

    def single_permutation(rng_local):
        if shuffling == 'labels1_based_on_2':
            if labels2 is None:
                raise ValueError("labels2 must be provided for 'labels1_based_on_2' shuffling")
            # Shuffle mapping at group level
            session2label1 = pd.Series(labels1).groupby(labels2).first()
            sessions = session2label1.index.values
            unique_vals = session2label1.values
            shuffled_vals = rng_local.permutation(unique_vals)
            new_mapping = dict(zip(sessions, shuffled_vals))
            return np.array([new_mapping[sess] for sess in labels2])
        
        elif shuffling == 'labels1_global':
            return rng_local.permutation(labels1)

        else:
            raise ValueError(f"Unknown shuffling mode: {shuffling}")

    if n_permut == 1:
        return single_permutation(rng), labels2
    else:
        seeds = rng.integers(0, 1e9, size=n_permut)
        permuted_labels1_list = Parallel(n_jobs=n_cores)(
            delayed(single_permutation)(np.random.default_rng(seed)) for seed in seeds
        )
        return permuted_labels1_list, labels2


# def fdr_correct_by_group(df, p_col='p', group_cols=None, alpha=0.05, method='fdr_bh'):
#    """
#     Apply FDR correction to p-values, optionally within groups.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Must include a p-value column.
#     p_col : str
#         Name of column with raw p-values.
#     group_cols : list or str or None
#         Grouping columns. If None, correct across whole DataFrame.
#     alpha : float
#         Significance level.
#     method : str
#         FDR method for `multipletests` (default 'fdr_bh').

#     Returns
#     -------
#     pd.DataFrame
#         Original df + ['p_corrected', 'reject'] columns.
#     """
#     if group_cols is None:
#         group_cols = []

#     def correct(group):
#         reject, p_corrected, _, _ = multipletests(group[p_col], alpha=alpha, method=method)
#         group = group.copy()
#         group['p_corrected'] = p_corrected
#         group['reject'] = reject
#         return group

#     if group_cols:
#         df_corrected = df.groupby(group_cols, group_keys=False).apply(correct)
#     else:
#         df_corrected = correct(df)

#     return df_corrected


def bf_gaussian_via_pearson(df: pd.DataFrame, y_col: str, x_col: str):
   """
    Bayes Factor for a simple linear relationship via Pearson r.

    Equivalent to testing β_x in y ~ x with Gaussian family + identity link.

    Parameters
    ----------
    df : pd.DataFrame
        Data.
    y_col : str
        Dependent variable column.
    x_col : str
        Independent variable column.

    Returns
    -------
    dict
        {'r': Pearson r, 'BF10': Bayes Factor}
    """
    sub = df[[y_col, x_col]].dropna()
    n = sub.shape[0]
    if n < 3:
        raise ValueError("Sample size too small (n<3) for Pearson BF.")
    
    r = pg.corr(sub[y_col], sub[x_col]).loc['pearson', 'r']

    bf10 = pg.bayesfactor_pearson(r=r, n=n)  # JZS prior r=0.707
    return {'r': float(r), 'BF10': float(bf10)}


def interpret_bayes_factor(bf):
    """
    Interpret Bayes Factor (BF10) with standard thresholds.

    Parameters
    ----------
    bf : float

    Returns
    -------
    str
        Qualitative label (e.g., 'strong H1', 'moderate H0').
    """
    try:
        bf = float(bf)
        if math.isnan(bf):
            return 'invalid BF'
    except Exception:
        return 'invalid BF'
    
    if bf > 10:
        return 'strong H1'
    elif bf > 3:
        return 'moderate H1'
    elif bf > 1:
        return 'weak H1'
    elif bf == 1:
        return 'inconclusive'
    elif bf > 1/3:
        return 'weak H0'
    elif bf > 1/10:
        return 'moderate H0'
    else:
        return 'strong H0'


def add_age_group(df):
    """
    Add age-related columns: categorical group + months + years.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'mouse_age' (days) or 'age_at_recording' (days).

    Returns
    -------
    pd.DataFrame
        Copy with new columns:
        - 'age_group' (young/old, by C.AGE_GROUP_THRESHOLD in days)
        - 'age_months'
        - 'age_years'
    """
    out = df.copy()

    if 'mouse_age' in out.columns:
        age = out['mouse_age']
    elif 'age_at_recording' in out.columns:
        age = out['age_at_recording']
    else:
        raise KeyError("Expected 'mouse_age' or 'mouse_Age_at_recording' in DataFrame.")

    out['age_group'] = (age > C.AGE_GROUP_THRESHOLD).map({True: 'old', False: 'young'})
    out['age_months'] = age / 30
    out['age_years'] = age / 365
    
    return out

