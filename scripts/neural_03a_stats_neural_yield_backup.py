# neural_04_stats_neural_yield.py

import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import Log
from utils.config import datapath, ROIs, align_event, trial_type
from utils.data_utils import shuffle_labels_perm, fdr_correct_by_group
from utils.permutation_test import plot_permut_test


def load_neural_yield_table():
    table = pd.read_parquet(
        os.path.join(datapath, f'ibl_BWMLL_neural_yield_{align_event}_{trial_type}_2025_full.parquet'))
    table['neural_yield'] = table['n_cluster'] / table['n_channel']
    table['age_group'] = table['age_at_recording'].map(lambda x: 'old' if x > 300 else 'young')
    table['age_years'] = table['age_at_recording'] / 365
    return table


def single_permutation(i, data, permuted_label, formula2use, family_func):
    try:
        shuffled_data = data.copy()
        shuffled_data['age_years'] = permuted_label
        model = glm(formula=formula2use, data=shuffled_data, family=family_func).fit()
        return model.params['age_years']
    except Exception as e:
        print(f"Permutation {i} failed: {e}")
        return np.nan

def run_permutation_analysis(table, y_var, region, formula2use, n_permut, family_func, n_jobs, shuffling, plot):
    region_data = table[table['Beryl_merge'] == region].dropna(subset=[y_var]).reset_index(drop=True)
    age_vals = region_data['age_years'].values

    permuted_labels1, _ = shuffle_labels_perm(
        labels1=age_vals,
        labels2=None,
        shuffling=shuffling,
        n_permut=n_permut,
        random_state=123,
        n_cores=4
    )

    null_dist = Parallel(n_jobs=n_jobs)(
        delayed(single_permutation)(i, region_data, permuted_labels1[i], formula2use, family_func)
        for i in tqdm(range(n_permut))
    )
    null_dist = np.array(null_dist)
    valid_null = null_dist[~np.isnan(null_dist)]

    model_obs = glm(formula=formula2use, data=region_data, family=family_func).fit()
    observed_val = model_obs.params['age_years']
    observed_val_p = model_obs.pvalues['age_years']
    p_perm = (np.sum(np.abs(valid_null) >= np.abs(observed_val)) + 1) / (len(valid_null) + 1)

    if plot:
        plot_permut_test(null_dist=valid_null, observed_val=observed_val, p=p_perm)

    return {
        'cluster_region': region,
        'y_var': y_var,
        'n_perm': n_permut,
        'formula': formula2use,
        'observed_val': observed_val,
        'observed_val_p': observed_val_p,
        'p_perm': p_perm,
        'ave_null_dist': valid_null.mean(),
        'null_dist': valid_null
    }





if __name__ == "__main__":

    n_jobs = 6
    n_permut = 10000
    shuffling = 'labels1_global'
    family_func = Gaussian()
    plot = True #TODO:
    save_results = True

    print("Loading neural yield table...")
    neural_yield_table = load_neural_yield_table()
    
    result_list_regions = []
    for y_var in ['n_cluster', 'neural_yield']:
        formula2use = f"{y_var} ~ age_years"
        for region in ROIs:
            print(f"Processing region {region}")
            result = run_permutation_analysis(
                neural_yield_table, y_var, region,
                formula2use, n_permut, family_func,
                n_jobs, shuffling, plot
            )
            result_list_regions.append(result)

    result_df = pd.DataFrame(result_list_regions)
    result_df = fdr_correct_by_group(result_df, p_col='p_perm', group_cols='y_var')

    if save_results:
        outpath = os.path.join(datapath, f'Neural_yield_ols_permut_{n_permut}permutation_{align_event}_{trial_type}_2025.csv')
        print(f"Saving results to {outpath}")
        result_df.drop(columns=['null_dist']).to_csv(outpath, index=False)

