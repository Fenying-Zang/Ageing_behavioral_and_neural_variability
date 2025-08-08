"""
permutation on neural yield

"""
#%%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ibl_style.style import figure_style
from ibl_style.utils import MM_TO_INCH
from utils.config import (datapath, figpath, align_event, trial_type, 
                          ROIs, palette, n_permut_behavior)
from utils.plot_utils import create_slice_org_axes
import figrid as fg
from utils.data_utils import shuffle_labels_perm
from joblib import Parallel, delayed
from statsmodels.genmod.families import Gaussian
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import Log 
from tqdm import tqdm
from utils.plot_utils import plot_permut_test
n_permut_behavior = 10

def load_neural_yield_table():
    table = pd.read_parquet(
        os.path.join(datapath, f'ibl_BWMLL_neural_yield_{align_event}_{trial_type}_2025_full.parquet'))
    table['neural_yield'] = table['n_cluster'] / table['n_channel']
    table['age_group'] = table['age_at_recording'].map(lambda x: 'old' if x > 300 else 'young')
    table['age_years'] = table['age_at_recording'] / 365
    return table

def single_permutation(i, data, permuted_label, formula2use, family_func=Gamma(link=Log())):
    try:
        shuffled_data = data.copy()
        shuffled_data['age_years'] = permuted_label

        model = glm(formula=formula2use, data=shuffled_data, family=family_func).fit()
        
        return model.params["age_years"]
    except Exception as e:
        print(f"Permutation {i} failed: {e}")
        return np.nan

def run_permutation_test(data, this_age, formula2use, family_func, n_permut, n_jobs):

    permuted_labels, _ = shuffle_labels_perm(
        labels1=this_age,
        labels2=None,
        shuffling='labels1_global',
        n_permut=n_permut,
        random_state=123,
        n_cores=n_jobs
    )

    null_dist = Parallel(n_jobs=n_jobs)(
        delayed(single_permutation)(i, data, permuted_labels[i], formula2use, family_func)
        for i in tqdm(range(n_permut))
    )


    null_dist = np.array(null_dist)
    valid_null = null_dist[~np.isnan(null_dist)]

    model_obs = glm(formula=formula2use, data=data, family=family_func).fit()
    observed_val = model_obs.params["age_years"]
    observed_val_p = model_obs.pvalues["age_years"]
    p_perm = (np.sum(np.abs(valid_null) >= np.abs(observed_val)) + 1) / (len(valid_null) + 1)

    return observed_val, observed_val_p, p_perm, valid_null



def mean(plot_permt_result=True):
    print("Loading data...")
    neural_yield_table = load_neural_yield_table()
    # stats = pd.read_csv(os.path.join(
    #     datapath, f'Neural_yield_ols_permut_1000permutation_{align_event}_{trial_type}_2025.csv'))

    for y_var in ['n_cluster', 'neural_yield']:
        # plot_yield_by_region(df, stats, y_var)
        shuffling='labels1_global'#'labels1_based_on_2'
        family_func = Gaussian()
        plot = True
        formula2use = f"{y_var} ~ age_years "
        region_results = []
        for region in ROIs:
            print(f'Processing region {region}')
            region_data = neural_yield_table[neural_yield_table['Beryl_merge']==region] 

            region_data = region_data[~ np.isnan(region_data[y_var])].reset_index(drop=True)
            this_data = region_data[y_var].values
            this_age = region_data['age_years'].values
            observed_val, observed_val_p, p_perm, valid_null = run_permutation_test(
                data=region_data,
                this_age=this_age,
                formula2use=formula2use,
                family_func=family_func,
                n_permut=n_permut_behavior,
                n_jobs=6
            )
            print(f"Omnibus results for {y_var}: \n  beta = {observed_val:.4f}, p_perm = {p_perm:.4f}")
            if plot_permt_result:
                plot_permut_test(null_dist=valid_null, observed_val=observed_val, p=p_perm, mark_p=None, metric=y_var, save_path=figpath, show=True, region=region)


            region_results.append({
                'cluster_region': region,
                'y_col': y_var,
                'n_perm': n_permut_behavior,
                'formula': formula2use,
                'observed_val': observed_val,
                'observed_val_p': observed_val_p,
                'p_perm': p_perm,
                'ave_null_dist': valid_null.mean()
            }) #  'null_dist': valid_nul

        pd.DataFrame(region_results).to_csv(
            datapath / f'Regional_{y_var}_{n_permut_behavior}permutation_2025.csv',
            index=False
        )



if __name__ == "__main__":

    mean()



# %%
