"""
Compute behavioral metrics (e.g., accuracy, bias, lapses) and run permutation tests.

Input:
    data/ibl_included_eids_trials_table2025_full.csv
Output:
    - Fit psychometric results:
        results/{split_type}_fit_psy_paras_age_info_{n_sessions}sessions_2025.csv
    - Permutation results:
        results/permutation_test_{label}_{age2use}_{n_permut}perm_2025.csv
"""
#%%
import numpy as np
import pandas as pd
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gaussian
import logging

# IBL libraries
from one.api import ONE
import brainbox as bb

# Project utils
import config as C
from scripts.utils.plot_utils import figure_style
from scripts.utils.stats_utils import run_permutation_test
from scripts.utils.behavior_utils import (
    compute_choice_history,
    fit_psychometric_paras,
    filter_trials
)
from scripts.utils.data_utils import (
    interpret_bayes_factor,
    bf_gaussian_via_pearson,
    add_age_group
)

log = logging.getLogger(__name__)

# === Load trials ===
# === Load trials table ===
def load_trial_table(filepath):
    """
    Load the pre-merged trials table and add choice history.

    Parameters
    ----------
    filepath : Path
        CSV file containing trial-level data.

    Returns
    -------
    pd.DataFrame
        Trials table with 'trialnum' and choice history columns.
    """
    try:
        trials_table = pd.read_csv(filepath)
        log.info(f"{len(set(trials_table['eid']))} sessions loaded")
        trials_table['trialnum'] = trials_table['trial_index']
        trials_table = compute_choice_history(trials_table)
        return trials_table
    except Exception as err:
        log.error(f"Error loading trials table: {err}")
        return None


# === Fit psychometric parameters and add RT variability results===
def fit_behavior(trials_table, trial_type=C.TRIAL_TYPE, clean_rt=True,
                 split_type='prevresp', save_results=True):
    """
    Filter trials and fit psychometric parameters, then merge with age info.

    Parameters
    ----------
    trials_table : pd.DataFrame
        Trial data across sessions.
    trial_type : str
        Which subset of trials to use (default from config).
    clean_rt : bool
        Whether to apply RT cleaning via filter_trials.
    split_type : str
        Split type passed to fit_psychometric_paras (e.g., 'prevresp').
    save_results : bool
        If True, save the resulting table.

    Returns
    -------
    pd.DataFrame
        Psychometric parameters, rt variability results per session with age information.
    """
    # Apply standard trial filters
    data2fit = filter_trials(
        trials_table, exclude_nan_event_trials=True, 
        trial_type=trial_type, event_list=C.EVENT_LIST, 
        clean_rt=clean_rt, rt_variable=C.RT_VARIABLE_NAME, 
        rt_cutoff=C.RT_CUTOFF)
    data2fit = add_age_group(data2fit)
    
    # Fit psychometric parameters, together compute rt variability
    fit_psy_paras = fit_psychometric_paras(data2fit, easy_trials=False, split_type=split_type)
    eid_unique = data2fit.drop_duplicates(subset='eid')
    
    # Merge with age info
    fit_psy_paras_age_info = pd.merge(
        fit_psy_paras,
        eid_unique[['eid', 'mouse_age', 'age_group']],
        on='eid', how='left'
    )
    fit_psy_paras_age_info= add_age_group(fit_psy_paras_age_info)
   
    # Save results
    if save_results:
        out_file = C.RESULTSPATH / f"{split_type}_fit_psy_paras_age_info_{len(eid_unique)}sessions_2025.csv"
        fit_psy_paras_age_info.to_csv(out_file, index=False)
        log.info(f"[Saved table] {out_file.resolve()}")
    return fit_psy_paras_age_info


# === Wrapper for permutation test across multiple measures ===
def run_permutation_for_measures(fit_data, measures_list, *, family_func, label,
                                 shuffling, n_permut, random_state, n_jobs=6,
                                 save=True, plot=True):
def run_permutation_for_measures(fit_data, measures_list, *, family_func, label,
                                 shuffling, n_permut, random_state, n_jobs=6,
                                 save=True, plot=True):
    """
    Run permutation tests for multiple behavioral measures against age.

    Parameters
    ----------
    fit_data : pd.DataFrame
        Session-level summary with measures and age info.
    measures_list : list of str
        Names of the dependent variables to test.
    family_func : statsmodels family
        GLM family (e.g. Gaussian()).
    label : str
        Label used for output filename.
    shuffling : str
        Shuffling scheme for permutation test.
    n_permut : int
        Number of permutations.
    random_state : int
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel workers.
    save : bool
        If True, save results to CSV.
    plot : bool
        If True, plot permutation distributions.

    Returns
    -------
    pd.DataFrame
        Results table with betas, p-values, Bayes factors.
    """    
    results = []
    for measure in measures_list:
        formula = f"{measure} ~ {C.AGE2USE}"
        subset = fit_data.dropna(subset=[measure]).reset_index(drop=True)
        age_labels = subset[C.AGE2USE].values

        # Bayes factor
        BF_dict = bf_gaussian_via_pearson(subset, measure, C.AGE2USE)

        # Permutation test (using utils)
        observed_val, observed_val_p, p_perm, null_dist = run_permutation_test(
            subset, age_labels,
            formula=formula, family_func=family_func,
            shuffling=shuffling, n_permut=n_permut,
            n_jobs=n_jobs, random_state=random_state,
            plot=plot
        )

        results.append({
            "y_var": measure,
            "formula": formula,
            "observed_val": observed_val,
            "observed_val_p": observed_val_p,
            "p_perm": p_perm,
            "pearson_r": BF_dict["r"],
            "BF10": BF_dict["BF10"],
            "BF_conclusion": interpret_bayes_factor(BF_dict["BF10"]),
            "n_perm": n_permut,
            "ave_null_dist": null_dist.mean() if null_dist.size > 0 else np.nan,
        })

        log.info(f"{measure}: β={observed_val:.3f}, p_perm={p_perm:.4f}, BF10={BF_dict['BF10']:.2f}")

    df_results = pd.DataFrame(results)
    if save:
        out_file = C.RESULTSPATH / f"permutation_test_{label}_{C.AGE2USE}_{n_permut}perm_2025.csv"
        df_results.to_csv(out_file, index=False)
        log.info(f"[Saved table] {out_file.resolve()}")

    return df_results

def main():
    """
    Main workflow:
      1. Load trials table
      2. Fit psychometric parameters and compute rt parameters (per session)
      3. Run permutation tests for psychometric, RT, and shift measures
    """
    trials_table_file = C.DATAPATH / 'ibl_included_eids_trials_table2025_full.csv'
    trials_table = load_trial_table(trials_table_file)

    if trials_table is not None:
        split_type='prevresp'
        
        # Step 1: Fit psychometric parameters, compute rt parameters and add age info
        fit_psy_paras_age_info = fit_behavior(
            trials_table, trial_type='first400', clean_rt=True,
            split_type=split_type, save_results=True
        ) #TODO:

        # Step 2: Define measure groups
        measures_list_psych = ['abs_bias', 'threshold', 'mean_lapse']
        measures_list_rt = ['rt_median', 'rt_CV']
        measures_list_shift = ['bias_shift', 'lapselow_shift', 'lapsehigh_shift']

        # Step 3: Run permutation tests for each group
        run_permutation_for_measures(
            fit_psy_paras_age_info, measures_list_psych,
            family_func=Gaussian(), label="psychometric",
            shuffling="labels1_global", n_permut=C.N_PERMUT_BEHAVIOR,
            random_state=C.RANDOM_STATE, n_jobs=6
        )

        run_permutation_for_measures(
            fit_psy_paras_age_info, measures_list_rt,
            family_func=Gaussian(), label="rt",
            shuffling="labels1_global", n_permut=C.N_PERMUT_BEHAVIOR,
            random_state=C.RANDOM_STATE, n_jobs=6
        )

        run_permutation_for_measures(
            fit_psy_paras_age_info, measures_list_shift,
            family_func=Gaussian(), label=f"{split_type}_shift",
            shuffling="labels1_global", n_permut=C.N_PERMUT_BEHAVIOR,
            random_state=C.RANDOM_STATE, n_jobs=6
        )


if __name__ == "__main__":
    from scripts.utils.io import setup_logging

    setup_logging()
    main()

