"""
Compute behavioral metrics (e.g., accuracy, bias, lapses)

Input:  data/ibl_included_eids_trials_table2025_full.csv # full trials table
Output: 
        #拟合psychometric结果
        results/{split_type}_fit_psy_paras_age_info_{len(eid_unique)}sessions_2025.csv 
        #置换检验结果
        results/permutation_test_{label}_{age2use}_{n_permut}perm_2025.csv 

"""
#%%

# === Imports ===
# === Standard Library ===
import os
import sys
import time
import pickle

# === Scientific Libraries ===
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gaussian, Gamma
from statsmodels.genmod.families.links import Log
from tqdm import tqdm
import pingouin as pg
# === IBL / Brainbox Libraries ===
from one.api import ONE
import brainbox as bb
from ibl_style.style import figure_style

# === Custom Project Utilities ===
from utils.config import (age2use,event_list,
    rt_variable_name, rt_cutoff, 
    age_group_threshold,datapath, figpath, align_event, 
    trial_type, ROIs, palette, age_group_threshold )

from utils.behavior_utils import (
    compute_choice_history,
    fit_psychometric_paras,
    filter_trials
)
# from utils.permutation_test import plot_permut_test
from utils.data_utils import shuffle_labels_perm, interpret_bayes_factor
from utils.plot_utils import plot_permut_test

# === Load trials ===
def load_trial_table(filepath):
    try:
        trials_table = pd.read_csv(filepath)
        print(len(set(trials_table['eid'])), 'sessions loaded')
        trials_table['trialnum'] = trials_table['trial_index']
        trials_table = compute_choice_history(trials_table)
        return trials_table
    except Exception as err:
        print(f'errored: {err}')
        return None

# === Fit psychometric parameters ===
def fit_behavior(trials_table, trial_type='first400', clean_rt=True,
                 split_type='prevresp', save_results=True):
    data2fit = filter_trials(
        trials_table, exclude_nan_event_trials=True, 
        trial_type=trial_type, event_list=event_list, 
        clean_rt=clean_rt, rt_variable=rt_variable_name, 
        rt_cutoff=rt_cutoff)
    data2fit['age_group'] = (data2fit['mouse_age'] > age_group_threshold).map({True: "old", False: "young"})

    fit_psy_paras = fit_psychometric_paras(data2fit, easy_trials=False, split_type=split_type)
    eid_unique = data2fit.drop_duplicates(subset='eid')

    fit_psy_paras_age_info = pd.merge(
        fit_psy_paras,
        eid_unique[['eid', 'mouse_age', 'age_group']],
        on='eid', how='left'
    )
    fit_psy_paras_age_info['age_months'] = fit_psy_paras_age_info['mouse_age'] / 30
    fit_psy_paras_age_info['age_years'] = fit_psy_paras_age_info['mouse_age'] / 365

    if save_results:
        out_path = os.path.join(datapath, f"{split_type}_fit_psy_paras_age_info_{len(eid_unique)}sessions_2025.csv")
        fit_psy_paras_age_info.to_csv(out_path, index=False)
    return fit_psy_paras_age_info


import pingouin as pg
import pandas as pd

def bf_gaussian_via_pearson(df: pd.DataFrame, y_col: str, x_col: str):
    """
    在 Gaussian + identity + 单自变量 场景下，
    用 Pearson r 的 Bayes 因子等价评估 y ~ 1 + x 中 x 的效应。
    
    参数
    ----
    df : pd.DataFrame
        含有 y 和 x 的数据表
    y_col : str
        因变量列名（metric）
    x_col : str
        自变量列名（age_years）
    
    返回
    ----
    dict: {
        'r': float,          # 皮尔逊相关
        'n': int,            # 样本量（去除缺失后）
        'BF10': float,       # 备择相对原假的证据
        'BF01': float        # 原假相对备择（= 1/BF10）
    }
    
    注意
    ----
    仅在 Gaussian family + identity link + 单一自变量 时，
    与 OLS 回归中 β_age 的检验完全等价。
    """
    sub = df[[y_col, x_col]].dropna()
    n = sub.shape[0]
    if n < 3:
        raise ValueError("样本量太小（n<3）无法计算 Pearson r 的 BF。")
    
    # r = pg.corr(sub[y_col], sub[x_col]).loc[0, 'r']
    r = pg.corr(sub[y_col], sub[x_col]).loc['pearson', 'r']

    bf10 = pg.bayesfactor_pearson(r=r, n=n)  # JZS 默认先验 r=0.707，双尾
    return {'r': float(r), 'n': int(n), 'BF10': float(bf10), 'BF01': float(1.0/bf10)}

# === Single permutation function ===
def single_permutation(i, data, permuted_label, formula2use, family_func):
    try:
        shuffled_data = data.copy()
        shuffled_data[age2use] = permuted_label
        model = glm(formula=formula2use, data=shuffled_data, family=family_func).fit()
        return model.params[age2use]
    except Exception as e:
        print(f"Permutation {i} failed: {e}")
        return np.nan

# === Perform permutation testing ===
def perf_permutation(measures_list, fit_data, shuffling, n_permut,
                     random_state=123, n_jobs=6, family_func=Gamma(link=Log()),
                     plot=True, save_results=True, label='psychometric'):
    results = []
    for measure in measures_list:
        formula2use = f"{measure} ~ {age2use}"
        data_subset = fit_data.dropna(subset=[measure]).reset_index(drop=True)
        this_age = data_subset[age2use].values
        this_metric = data_subset[measure].values
        BF_dict = bf_gaussian_via_pearson(data_subset, measure, age2use)
        print(f"BF10 for {measure} vs. {age2use}: {BF_dict['BF10']:.3f}, r={BF_dict['r']:.3f}, n={BF_dict['n']}")

        permuted_labels, _ = shuffle_labels_perm(
            labels1=this_age, labels2=None, shuffling=shuffling,
            n_permut=n_permut, random_state=random_state, n_cores=n_jobs
        )

        null_dist = Parallel(n_jobs=n_jobs)(
            delayed(single_permutation)(i, data_subset, permuted_labels[i], formula2use, family_func)
            for i in tqdm(range(n_permut))
        )
        null_dist = np.array(null_dist)
        valid_null = null_dist[~np.isnan(null_dist)]

        model_obs = glm(formula=formula2use, data=data_subset, family=family_func).fit()
        observed_val = model_obs.params[age2use]
        observed_val_p = model_obs.pvalues[age2use]
        p_perm = (np.sum(np.abs(valid_null) >= np.abs(observed_val)) + 1) / (len(valid_null) + 1)

        if plot:
            plot_permut_test(null_dist=valid_null, observed_val=observed_val, p=p_perm)

        print(f"{measure}: observed={observed_val:.3f}, perm_p={p_perm:.4f}")
        results.append({
            'y_var': measure,
            'n_perm': n_permut,
            'formula': formula2use,
            'observed_val': observed_val,
            'observed_val_p': observed_val_p,
            'p_perm': p_perm,
            'person_r': BF_dict['r'],
            'BF10': BF_dict['BF10'],  # Placeholder for Bayes Factor if needed
            'BF_conclusion': interpret_bayes_factor(BF_dict['BF10']),
            'ave_null_dist': valid_null.mean(),
            'null_dist': valid_null
        })

    result_df = pd.DataFrame(results)
    if save_results:
        out_file = os.path.join(datapath, f'permutation_test_{label}_{age2use}_{n_permut}perm_2025.csv')
        result_df.to_csv(out_file, index=False)
    return result_df

# === Main execution ===
if __name__ == "__main__":
    trials_table_file = os.path.join(datapath, 'ibl_included_eids_trials_table2025_full.csv')
    trials_table = load_trial_table(trials_table_file)

    if trials_table is not None:
        # Fit psychometric parameters
        fit_psy_paras_age_info = fit_behavior(
            trials_table, trial_type='first400', clean_rt=True,
            split_type='prevresp', save_results=True
        ) #TODO:

        # Define measure lists
        measures_list_psych = ['abs_bias', 'threshold', 'mean_lapse']
        measures_list_rt = ['rt_median', 'rt_CV']
        measures_list_shift = ['bias_shift', 'lapselow_shift', 'lapsehigh_shift']

        # Run permutation test for each group
        perf_permutation(measures_list_psych, fit_psy_paras_age_info,
                         shuffling='labels1_global', n_permut=10000,
                         random_state=123, n_jobs=6, family_func=Gaussian(),
                         plot=True, save_results=True, label='psychometric')

        perf_permutation(measures_list_rt, fit_psy_paras_age_info,
                         shuffling='labels1_global', n_permut=10000,
                         random_state=123, n_jobs=6, family_func=Gaussian(),
                         plot=True, save_results=True, label='rt')

        perf_permutation(measures_list_shift, fit_psy_paras_age_info,
                         shuffling='labels1_global', n_permut=10000,
                         random_state=123, n_jobs=6, family_func=Gaussian(),
                         plot=True, save_results=True, label='shift')
