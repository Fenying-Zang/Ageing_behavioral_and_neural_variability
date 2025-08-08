"""
neural yield; 7 metrics

"""
#%%#
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gamma, Gaussian
from statsmodels.genmod.families.links import Log

from utils.config import (
    datapath, figpath, align_event, trial_type, ROIs,
    age_group_threshold, metrics_with_meansub, metrics_without_meansub,
    # n_permut_neural_omnibus, n_permut_neural_regional #TODO:
)
# from utils.permutation_test import shuffle_labels, plot_permut_test #check 
from utils.plot_utils import plot_permut_test
from joblib import parallel_backend


def load_neural_metrics_data(df_path):
    """
    Load extracted metrics summary data
    """
    return pd.read_parquet(df_path)


def get_suffix(mean_subtraction):
    return 'meansub' if mean_subtraction else ''


def get_link_func(metric):
    if metric in ['ff_quench', 'fr_delta_modulation', 'ff_quench_modulation']:
        return Gaussian()
    else:
        # return Gamma(link=Log())
        return Gaussian() #TODO:

def single_permutation(i, data, permuted_label, formula2use, family_func):
    try:
        shuffled_data = data.copy()
        shuffled_data['age_years'] = permuted_label

        model = glm(formula=formula2use, data=shuffled_data, family=family_func).fit()
        return model.params["age_years"]
    except Exception as e:
        print(f"Permutation {i} failed: {e}")
        return np.nan
    

def shuffle_labels(labels1, labels2, n_permut=1, shuffling='labels1_based_on_2', n_cores=1, random_state=None):
    """
    Shuffle labels1 (e.g., age) based on labels2 (e.g., session), preserving group-level label sharing.
    
    Parameters:
        labels1: array-like, e.g., age per row
        labels2: array-like, e.g., session per row
        n_permut: int, number of permutations
        shuffling: str, must be 'labels1_based_on_2'
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
    labels2 = np.array(labels2)
    
    if shuffling != 'labels1_based_on_2':
        raise ValueError("Currently only supports 'labels1_based_on_2' shuffling")

    # Mapping: session → age
    session2age = pd.Series(labels1).groupby(labels2).first()
    sessions = session2age.index.values
    unique_ages = session2age.values

    # Set up RNG
    rng = np.random.default_rng(seed=random_state)

    def single_permutation(rng_local):
        shuffled_ages = rng_local.permutation(unique_ages)
        new_mapping = dict(zip(sessions, shuffled_ages))
        return np.array([new_mapping[sess] for sess in labels2])

    if n_permut == 1:
        return single_permutation(rng), labels2
    else:
        # Use different seeds for each job to ensure independence
        seeds = rng.integers(0, 1e9, size=n_permut)
        permuted_labels1_list = Parallel(n_jobs=n_cores)(
            delayed(single_permutation)(np.random.default_rng(seed)) for seed in seeds
        )
        return permuted_labels1_list, labels2


def def_glm_formula(metric, mean_subtraction=False,log_transform=False):
    # 定义模型公式 TODO:
    if metric in [ 'fr_delta_modulation', 'ff_quench_modulation']:
        formula2use = f"{metric} ~ age_years + C(cluster_region) + n_trials" 
    elif metric in ['ff_quench']:
        if mean_subtraction:
            formula2use = f"{metric} ~ age_years + C(cluster_region) + n_trials"
        else:
            formula2use = f"{metric} ~ age_years + C(cluster_region) + abs_contrast + n_trials"

    else:
        if mean_subtraction:
            if log_transform:
                formula2use = f"log_{metric} ~ age_years + C(cluster_region) + n_trials"
            else:
                formula2use = f"{metric} ~ age_years + C(cluster_region) + n_trials"

        else:
            if log_transform:
                formula2use = f"log_{metric} ~ age_years + C(cluster_region) + abs_contrast + n_trials"
            else:
                formula2use = f"{metric} ~ age_years + C(cluster_region) + abs_contrast + n_trials"

    return formula2use

def run_permutation_test(data, this_age, this_eid, formula2use, family_func, n_permut, n_jobs):

    permuted_labels, _ = shuffle_labels(
        labels1=this_age,
        labels2=this_eid,
        shuffling='labels1_based_on_2',
        n_permut=n_permut,
        random_state=123,
        n_cores=n_jobs
    )
    with parallel_backend('loky'):
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

def main(mean_subtraction=False, plot_permt_result=True, log_transform=False):
    metrics_path = datapath / ("neural_metrics_summary_meansub_merged.parquet" if mean_subtraction
                                else "neural_metrics_summary_conditions_merged.parquet")
    selected_metrics = metrics_with_meansub if mean_subtraction else metrics_without_meansub

    print("Loading extracted neural metrics summary...")
    neural_metrics = pd.read_parquet(metrics_path)
    neural_metrics['age_group'] = neural_metrics['mouse_age'].map(lambda x: 'old' if x > age_group_threshold else 'young')
    neural_metrics['mouse_age_months'] = neural_metrics['mouse_age'] / 30
    neural_metrics['age_years'] = neural_metrics['mouse_age'] / 365
    # neural_metrics['pre_frs'] = neural_metrics['pre_frs']+ 1e-6
    neural_metrics['log_pre_fr'] = np.log(neural_metrics['pre_fr']+ 1e-6)
    neural_metrics['log_post_fr'] = np.log(neural_metrics['post_fr']+ 1e-6)

    # neural_metrics['pre_FFs'] = neural_metrics['pre_FFs'] + 1e-6
    neural_metrics['log_pre_ff'] = np.log(neural_metrics['pre_ff']+ 1e-6)
    neural_metrics['log_post_ff'] = np.log(neural_metrics['post_ff']+ 1e-6)

    # for metric, _ in selected_metrics:
    for metric in ['fr_delta_modulation', 'ff_quench_modulation']:

        print(f"\nRunning permutation test for {metric}...")
        neural_metrics2use = neural_metrics[~np.isnan(neural_metrics[metric])].reset_index(drop=True)
        
        #TODO: for CM, drop duplicate rows
        if metric in ['fr_delta_modulation', 'ff_quench_modulation']:
            neural_metrics2use = neural_metrics2use.drop(columns=['signed_contrast', 'abs_contrast'], errors='ignore')
            neural_metrics2use = neural_metrics2use.drop_duplicates(subset=['uuids', 'cluster_region','session_pid', 'age_years', 'age_group', metric])
            print(len(neural_metrics2use), "rows after dropping duplicates for CM metrics")
        this_age = neural_metrics2use['age_years'].values
        this_eid = neural_metrics2use['session_eid'].values
        family_func = get_link_func(metric)
        formula2use = def_glm_formula(metric, mean_subtraction, log_transform)


        # Omnibus test
        observed_val, observed_val_p, p_perm, valid_null = run_permutation_test(
            data=neural_metrics2use,
            this_age=this_age,
            this_eid=this_eid,
            formula2use=formula2use,
            family_func=family_func,
            n_permut=n_permut_neural_omnibus,
            n_jobs=6
        )

        print(f"Omnibus results for {metric}: \n  beta = {observed_val:.4f}, p_perm = {p_perm:.4f}")
        if plot_permt_result:
            plot_permut_test(null_dist=valid_null, observed_val=observed_val, p=p_perm, mark_p=None)

        pd.DataFrame([{
            'y_col': metric,
            'n_perm': n_permut_neural_omnibus,
            'formula': formula2use,
            'observed_val': observed_val,
            'observed_val_p': observed_val_p,
            'p_perm': p_perm,
            'ave_null_dist': valid_null.mean()
        }]).to_csv(
            datapath / f'Omnibus_{metric}_{n_permut_neural_omnibus}permutation_{align_event}_{trial_type}_{get_suffix(mean_subtraction)}_2025_vNEW_log.csv',
            index=False
        )  #  'null_dist': valid_nul

        # Region-specific test
        region_results = []
        for region in ROIs:
            region_data = neural_metrics2use[neural_metrics2use['cluster_region'] == region]
            if region_data.empty:
                continue
            print(f"Processing region: {region}")
            this_age = region_data['age_years'].values
            this_eid = region_data['session_eid'].values
            observed_val, observed_val_p, p_perm, valid_null = run_permutation_test(
                data=region_data,
                this_age=this_age,
                this_eid=this_eid,
                formula2use=formula2use,
                family_func=family_func,
                n_permut=n_permut_neural_regional,
                n_jobs=6
            )

            if plot_permt_result:
                # plot_permut_test(null_dist=valid_null, observed_val=observed_val, p=p_perm, mark_p=None)
                plot_permut_test(null_dist=valid_null, observed_val=observed_val, p=p_perm, mark_p=None, metric=metric, save_path=figpath, show=True, region=region)

            region_results.append({
                'cluster_region': region,
                'y_col': metric,
                'n_perm': n_permut_neural_regional,
                'formula': formula2use,
                'observed_val': observed_val,
                'observed_val_p': observed_val_p,
                'p_perm': p_perm,
                'ave_null_dist': valid_null.mean()
            }) #  'null_dist': valid_nul

        pd.DataFrame(region_results).to_csv(
            datapath / f'Regional_{metric}_{n_permut_neural_regional}permutation_{align_event}_{trial_type}_{get_suffix(mean_subtraction)}_2025_vNEW_log.csv',
            index=False
        )


if __name__ == "__main__":
    n_permut_neural_omnibus =1000
    n_permut_neural_regional=1000

    main(mean_subtraction=False, plot_permt_result=True, log_transform=True)
# %%
