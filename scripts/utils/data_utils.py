import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os
import pingouin as pg
import math
import config as C


def load_filtered_recordings(datapath=C.DATAPATH, filename = 'BWM_LL_release_afterQC_df.csv'):
    """
    
    paras:
    matrix -- filename and position
    
    return:
    result -- filtered recordings
    """
    try:
        recordings_filtered = pd.read_csv( datapath / f"{filename}" )
    except Exception as err:
        print(f'errored: {err}')
        recordings_filtered =np.nan
    return recordings_filtered


def normalize_units(matrix):
    """
    
    paras:
    matrix --  (trials, units, timepoints) array
    
    return:
    result -- normalized (trials, units, timepoints) array
    """
    trials, units, timepoints = matrix.shape
    
    result = np.zeros_like(matrix)
    # normalized_data
    
    for i in range(units):
        data = matrix[:, i, :].flatten()
        
        
        min_val = np.min(data)
        max_val = np.max(data)

        # max-min normalization if max_val> min_val
        if max_val > min_val:
            result[:, i, :] = (matrix[:, i, :] - min_val) / (max_val - min_val)

        else:
            result[:, i, :] = np.zeros_like(matrix[:, i, :]) 
    
    return result


def shuffle_labels_perm(labels1, labels2, n_permut=1, shuffling='labels1_based_on_2', n_cores=1, random_state=None):
    """
    Shuffle labels1 (e.g., age) based on labels2 (e.g., session), preserving group-level label sharing.
    
    Parameters:
        labels1: array-like, e.g., age per row
        labels2: array-like, e.g., session per row
        n_permut: int, number of permutations
        shuffling: str, 'labels1_based_on_2' or 'labels1'
        n_cores: int, number of cores for parallel processing
        random_state: int or None, random seed for reproducibility

    Returns:
        If n_permut == 1:
            permuted_labels1: array
            labels2: unchanged
        Else:
            permuted_labels1_list: list of arrays
            labels2: unchanged
    """
    labels1 = np.array(labels1)
    if labels2 is not None:
        labels2 = np.array(labels2)


    rng = np.random.default_rng(seed=random_state)

    def single_permutation(rng_local):
        if shuffling == 'labels1_based_on_2':
            if labels2 is None:
                raise ValueError("labels2 must be provided for 'labels1_based_on_2' shuffling")

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


def fdr_correct_by_group(df, p_col='p', group_cols=None, alpha=0.05, method='fdr_bh'):
    """
    对 DataFrame 中按 group_cols 分组的 p 值列进行 FDR 校正。

    参数：
        df: pd.DataFrame
        p_col: str，表示存放原始 p 值的列名
        group_cols: list 或 str，要分组的列名（可为空）
        alpha: 显著性水平
        method: FDR 校正方法，默认 'fdr_bh'

    返回：
        pd.DataFrame，增加了 'p_corrected' 和 'reject' 列
    """
    if group_cols is None:
        group_cols = []

    def correct(group):
        reject, p_corrected, _, _ = multipletests(group[p_col], alpha=alpha, method=method)
        group = group.copy()
        group['p_corrected'] = p_corrected
        group['reject'] = reject
        return group

    if group_cols:
        df_corrected = df.groupby(group_cols, group_keys=False).apply(correct)
    else:
        df_corrected = correct(df)

    return df_corrected



def bf_gaussian_via_pearson(df: pd.DataFrame, y_col: str, x_col: str):
    """
    在 Gaussian + identity + 单自变量 场景下，
    用 Pearson r 的 Bayes 因子等价评估 y ~ 1 + x 中 x 的效应。
    

    ----
    df : pd.DataFrame
    y_col : str metric
    x_col : str
    
    ----
    dict: {
        'r': float,          # 皮尔逊相关
        'BF10': float,       # 备择相对原假的证据
    }
    
    ----
    仅在 Gaussian family + identity link + 单一自变量 时，
    与 OLS 回归中 β_age 的检验完全等价。
    """
    sub = df[[y_col, x_col]].dropna()
    n = sub.shape[0]
    if n < 3:
        raise ValueError("样本量太小（n<3）无法计算 Pearson r 的 BF。")
    
    r = pg.corr(sub[y_col], sub[x_col]).loc['pearson', 'r']

    bf10 = pg.bayesfactor_pearson(r=r, n=n)  # JZS 默认先验 r=0.707，双尾
    return {'r': float(r), 'BF10': float(bf10)}


def interpret_bayes_factor(bf):
    try:
        bf = float(bf)
        if math.isnan(bf):
            return 'invalid BF'
    except:
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
    """Return a copy with a categorical 'age_group' column based on C.AGE_GROUP_THRESHOLD.
    Accepts age column named either 'mouse_age' or 'mouse_Age_at_recording' (in days).
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



    # # Mapping: session → age
    # session2age = pd.Series(labels1).groupby(labels2).first()
    # sessions = session2age.index.values
    # unique_ages = session2age.values

    # # Set up RNG
    # rng = np.random.default_rng(seed=random_state)

    # def single_permutation(rng_local):
    #     shuffled_ages = rng_local.permutation(unique_ages)
    #     new_mapping = dict(zip(sessions, shuffled_ages))
    #     return np.array([new_mapping[sess] for sess in labels2])

    # if n_permut == 1:
    #     return single_permutation(rng), labels2
    # else:
    #     # Use different seeds for each job to ensure independence
    #     seeds = rng.integers(0, 1e9, size=n_permut)
    #     permuted_labels1_list = Parallel(n_jobs=n_cores)(
    #         delayed(single_permutation)(np.random.default_rng(seed)) for seed in seeds
    #     )
    #     return permuted_labels1_list, labels2
