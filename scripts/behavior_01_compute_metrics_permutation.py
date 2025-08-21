"""
Compute behavioral metrics (e.g., accuracy, bias, lapses)

Input:  data/ibl_included_eids_trials_table2025_full.csv # full trials table


"""
#%%
import os
import sys
import time
import pickle
import numpy as np
import pingouin as pg
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gaussian, Gamma
from statsmodels.genmod.families.links import Log
from tqdm import tqdm
# === IBL / Brainbox Libraries ===
from one.api import ONE
import brainbox as bb
from scripts.utils.plot_utils import figure_style

# === Custom Project Utilities ===
import config as C
from scripts.utils.behavior_utils import (
    compute_choice_history,
    fit_psychometric_paras,
    filter_trials
)
from scripts.utils.data_utils import (
    shuffle_labels_perm, 
    interpret_bayes_factor, 
    bf_gaussian_via_pearson
)
from scripts.utils.plot_utils import plot_permut_test


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
def fit_behavior(trials_table, trial_type=C.TRIAL_TYPE, clean_rt=True,
                 split_type='prevresp', save_results=True):
    data2fit = filter_trials(
        trials_table, exclude_nan_event_trials=True, 
        trial_type=trial_type, event_list=C.EVENT_LIST, 
        clean_rt=clean_rt, rt_variable=C.RT_VARIABLE_NAME, 
        rt_cutoff=C.RT_CUTOFF)
    data2fit['age_group'] = (data2fit['mouse_age'] > C.AGE_GROUP_THRESHOLD).map({True: "old", False: "young"})

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
        out_path = os.path.join(C.RESULTSPATH, f"{split_type}_fit_psy_paras_age_info_{len(eid_unique)}sessions_2025.csv")
        fit_psy_paras_age_info.to_csv(out_path, index=False)
    return fit_psy_paras_age_info


# === Single permutation function ===
def single_permutation(i, data, permuted_label, formula2use, family_func):
    try:
        shuffled_data = data.copy()
        shuffled_data[C.AGE2USE] = permuted_label
        model = glm(formula=formula2use, data=shuffled_data, family=family_func).fit()
        return model.params[C.AGE2USE]
    except Exception as e:
        print(f"Permutation {i} failed: {e}")
        return np.nan

# === Perform permutation testing ===
def perf_permutation(measures_list, fit_data, shuffling, n_permut,
                     random_state=123, n_jobs=6, family_func=Gamma(link=Log()),
                     plot=True, save_results=True, label='psychometric'):
    results = []
    for measure in measures_list:
        formula2use = f"{measure} ~ {C.AGE2USE}"
        data_subset = fit_data.dropna(subset=[measure]).reset_index(drop=True)
        this_age = data_subset[C.AGE2USE].values
        this_metric = data_subset[measure].values
        BF_dict = bf_gaussian_via_pearson(data_subset, measure, C.AGE2USE)
        print(f"BF10 for {measure} vs. {C.AGE2USE}: {BF_dict['BF10']:.3f}, r={BF_dict['r']:.3f}, n={BF_dict['n']}")

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
        observed_val = model_obs.params[C.AGE2USE]
        observed_val_p = model_obs.pvalues[C.AGE2USE]
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
            'BF10': BF_dict['BF10'], 
            'BF_conclusion': interpret_bayes_factor(BF_dict['BF10']),
            'ave_null_dist': valid_null.mean(),
            'null_dist': valid_null
        })

    result_df = pd.DataFrame(results)
    if save_results:
        out_file = os.path.join(C.DATAPATH, f'permutation_test_{label}_{C.AGE2USE}_{n_permut}perm_2025.csv')
        result_df.to_csv(out_file, index=False)
    return result_df

# === Main execution ===
if __name__ == "__main__":
    trials_table_file = os.path.join(C.DATAPATH, 'ibl_included_eids_trials_table2025_full.csv')
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
                         shuffling='labels1_global', n_permut=C.N_PERMUT_BEHAVIOR,
                         random_state=C.RANDOM_STATE, n_jobs=6, family_func=Gaussian(),
                         plot=True, save_results=True, label='psychometric')

        perf_permutation(measures_list_rt, fit_psy_paras_age_info,
                         shuffling='labels1_global', n_permut=C.N_PERMUT_BEHAVIOR,
                         random_state=C.RANDOM_STATE, n_jobs=6, family_func=Gaussian(),
                         plot=True, save_results=True, label='rt')

        perf_permutation(measures_list_shift, fit_psy_paras_age_info,
                         shuffling='labels1_global', n_permut=C.N_PERMUT_BEHAVIOR,
                         random_state=C.RANDOM_STATE, n_jobs=6, family_func=Gaussian(),
                         plot=True, save_results=True, label='shift')
