"""
neural yield; 7 metrics

"""
#%%#
import pandas as pd
import numpy as np
from statsmodels.genmod.families import Gaussian
import config as C
from scripts.utils.plot_utils import plot_permut_test
from scripts.utils.io import read_table, get_suffix
from scripts.utils.stats_utils import run_permutation_test  
from scripts.utils.io import read_table
import logging


log = logging.getLogger(__name__)
FAMILY_FUNC = Gaussian()
N_JOBS = 6
SHUFFLING = 'labels1_based_on_2'  # same as your other scripts


def def_glm_formula(metric, mean_subtraction=False, log_transform=False):
    """Return GLM formula string for a metric, switching covariates by metric & mean_subtraction/log_transform."""
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


def drop_term_from_formula(formula, term="C(cluster_region)"):
    """Remove a single RHS term from a Patsy-style formula safely."""
    lhs, rhs = [s.strip() for s in formula.split("~", 1)]
    terms = [t.strip() for t in rhs.split("+")]
    terms = [t for t in terms if t != term and t != ""]
    rhs_new = " + ".join(terms) if terms else "1"
    return f"{lhs} ~ {rhs_new}"


def main(mean_subtraction=False, plot_permt_result=True, log_transform=False):
    """Load neural metrics → clean/log-transform → (omnibus + per-region) permutation with session-grouped shuffling → save CSVs."""

    metrics_path = C.DATAPATH / ("neural_metrics_summary_meansub.parquet" if mean_subtraction
                                else "neural_metrics_summary_conditions.parquet")
    selected_metrics = C.METRICS_WITH_MEANSUB if mean_subtraction else C.METRICS_WITHOUT_MEANSUB

    print("Loading extracted neural metrics summary...")
    neural_metrics = read_table(metrics_path)
    neural_metrics['age_group'] = neural_metrics['mouse_age'].map(lambda x: 'old' if x > C.AGE_GROUP_THRESHOLD else 'young')
    neural_metrics['mouse_age_months'] = neural_metrics['mouse_age'] / 30
    neural_metrics['age_years'] = neural_metrics['mouse_age'] / 365
    # neural_metrics['pre_frs'] = neural_metrics['pre_frs']+ 1e-6
    neural_metrics['log_pre_fr'] = np.log(neural_metrics['pre_fr']+ 1e-6)
    neural_metrics['log_post_fr'] = np.log(neural_metrics['post_fr']+ 1e-6)

    # neural_metrics['pre_FFs'] = neural_metrics['pre_FFs'] + 1e-6
    neural_metrics['log_pre_ff'] = np.log(neural_metrics['pre_ff']+ 1e-6)
    neural_metrics['log_post_ff'] = np.log(neural_metrics['post_ff']+ 1e-6)

    for metric, _ in selected_metrics: #TODO:
    # for metric in ['fr_delta_modulation', 'ff_quench_modulation']:

        print(f"\nRunning permutation test for {metric}...")
        neural_metrics2use = neural_metrics[~np.isnan(neural_metrics[metric])].reset_index(drop=True)
        
        #for CM, drop duplicate rows
        if metric in ['fr_delta_modulation', 'ff_quench_modulation']:
            neural_metrics2use = neural_metrics2use.drop(columns=['signed_contrast', 'abs_contrast'], errors='ignore')
            neural_metrics2use = neural_metrics2use.drop_duplicates(subset=['uuids', 'cluster_region','session_pid', 'age_years', 'age_group', metric])
            print(len(neural_metrics2use), "rows after dropping duplicates for CM metrics")
        this_age = neural_metrics2use['age_years'].values
        this_eid = neural_metrics2use['session_eid'].values
        # family_func = FAMILY_FUNC
        formula_full = def_glm_formula(metric, mean_subtraction, log_transform)

        observed_val, observed_val_p, p_perm, valid_null = run_permutation_test(
            data=neural_metrics2use,
            age_labels=this_age,                
            group_labels=this_eid,               
            formula=formula_full,
            family_func=FAMILY_FUNC,
            shuffling=SHUFFLING,                 # 'labels1_based_on_2'
            n_permut=C.N_PERMUT_NEURAL_OMNIBUS, 
            n_jobs=N_JOBS,
            random_state=C.RANDOM_STATE,
            plot=False
        )

        print(f"Omnibus results for {metric}: \n  beta = {observed_val:.4f}, p_perm = {p_perm:.4f}")
        if plot_permt_result:
            plot_permut_test(null_dist=valid_null, observed_val=observed_val, p=p_perm, mark_p=None)

        pd.DataFrame([{
            'y_col': metric,
            'n_perm': C.N_PERMUT_NEURAL_OMNIBUS,
            'formula': formula_full,
            'observed_val': observed_val,
            'observed_val_p': observed_val_p,
            'p_perm': p_perm,
            'ave_null_dist': valid_null.mean()
        }]).to_csv(
            C.RESULTSPATH / f'Omnibus_{metric}_{C.N_PERMUT_NEURAL_OMNIBUS}permutation_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_{get_suffix(mean_subtraction)}.csv',
            index=False
        )  #  'null_dist': valid_nul

        # Region-specific test
        region_results = []
        for region in C.ROIS:
            region_data = neural_metrics2use[neural_metrics2use['cluster_region'] == region]
            if region_data.empty:
                continue
            print(f"Processing region: {region}")
            this_age = region_data['age_years'].values
            this_eid = region_data['session_eid'].values

            formula_region = drop_term_from_formula(formula_full, "C(cluster_region)")
            observed_val, observed_val_p, p_perm, valid_null = run_permutation_test(
                data=region_data,
                age_labels=this_age,                
                group_labels=this_eid,              
                formula=formula_region,
                family_func=FAMILY_FUNC,
                shuffling=SHUFFLING,                 # 'labels1_based_on_2'
                n_permut=C.N_PERMUT_NEURAL_REGIONAL,  
                n_jobs=N_JOBS,
                random_state=C.RANDOM_STATE,
                plot=False
            )


            if plot_permt_result:
                plot_permut_test(null_dist=valid_null, observed_val=observed_val, p=p_perm, mark_p=None, metric=metric, save_path=C.FIGPATH, show=True, region=region)

            region_results.append({
                'cluster_region': region,
                'y_col': metric,
                'n_perm': C.N_PERMUT_NEURAL_REGIONAL,
                'formula': formula_region,
                'observed_val': observed_val,
                'observed_val_p': observed_val_p,
                'p_perm': p_perm,
                'ave_null_dist': valid_null.mean()
            }) #  'null_dist': valid_nul

        outfile = C.RESULTSPATH / f'Regional_{metric}_{C.N_PERMUT_NEURAL_REGIONAL}permutation_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_{get_suffix(mean_subtraction)}.csv'
        pd.DataFrame(region_results).to_csv(outfile, index=False)
        log.info(f"[Saved table] {outfile.resolve()}")


if __name__ == "__main__":

    from scripts.utils.io import setup_logging
    setup_logging() 

    for mean_sub in (True, False):
        logging.info(f"=== Run with mean_subtraction={mean_sub} ===")
        main(mean_subtraction=mean_sub,
             plot_permt_result=False,
             log_transform=True)

# %%
