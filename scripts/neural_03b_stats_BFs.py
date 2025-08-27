"""
compute BFs with BayesFactor package, without any random effects structure
"""
#%%
import pandas as pd
import numpy as np
import os
import platform
from scripts.utils.data_utils import interpret_bayes_factor, add_age_group
from scripts.utils.io import read_table, get_suffix

import config as C
from rpy2.robjects import Formula
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter    
from rpy2.robjects import default_converter, pandas2ri  
import logging

log = logging.getLogger(__name__)
# ---------- R env ----------
os.environ['R_HOME'] = r"C:/PROGRA~1/R/R-44~1.1" #TODO: costum your own R path here
os.environ['R_USER'] = os.path.expanduser("~")
os.environ["R_DISABLE_CONSOLE_OUTPUT"] = "TRUE"
os.environ['RPY2_CFFI_MODE'] = 'ABI'  

# Load BayesFactor 
bayesfactor = importr('BayesFactor')

# Disable JIT (avoids rpy2 issues on Windows)
ro.r('compiler::enableJIT(0)')

try:  
    if platform.system() == "Windows":
        ro.r('Sys.setlocale("LC_CTYPE", "English_United States")')
    else:
        ro.r('Sys.setlocale("LC_ALL", "English_United_States.UTF-8")')
except Exception:
    pass  


# ---------- helpers ----------
def def_BF_formula(metric, mean_subtraction=False, log_transform=False):
    if metric in ['fr_delta_modulation', 'ff_quench_modulation']:
        formula_full_str = f'{metric} ~ age_years + cluster_region + n_trials'
        formula_reduced_age_str = f'{metric} ~ cluster_region + n_trials'
        formula_reduced_region_str = f'{metric} ~ age_years + n_trials'
    elif metric in ['ff_quench']:
        if mean_subtraction:
            formula_full_str = f'{metric} ~ age_years + n_trials + cluster_region'
            formula_reduced_age_str = f'{metric} ~ n_trials + cluster_region'
            formula_reduced_region_str = f'{metric} ~ age_years + n_trials'
        else:
            formula_full_str = f'{metric} ~ age_years + abs_contrast + n_trials + cluster_region'
            formula_reduced_age_str = f'{metric} ~ abs_contrast + n_trials + cluster_region'
            formula_reduced_region_str = f'{metric} ~ age_years + abs_contrast + n_trials'
    else:
        if mean_subtraction:
            if log_transform:
                formula_full_str = f'log_{metric} ~ age_years + n_trials + cluster_region'
                formula_reduced_age_str = f'log_{metric} ~ n_trials + cluster_region'
                formula_reduced_region_str = f'log_{metric} ~ age_years + n_trials'
            else:
                formula_full_str = f'{metric} ~ age_years + n_trials + cluster_region'
                formula_reduced_age_str = f'{metric} ~ n_trials + cluster_region'
                formula_reduced_region_str = f'{metric} ~ age_years + n_trials'
        else:
            if log_transform:
                formula_full_str = f'log_{metric} ~ age_years + abs_contrast + n_trials + cluster_region'
                formula_reduced_age_str = f'log_{metric} ~ abs_contrast + n_trials + cluster_region'
                formula_reduced_region_str = f'log_{metric} ~ age_years + abs_contrast + n_trials'
            else:
                formula_full_str = f'{metric} ~ age_years + abs_contrast + n_trials + cluster_region'
                formula_reduced_age_str = f'{metric} ~ abs_contrast + n_trials + cluster_region'
                formula_reduced_region_str = f'{metric} ~ age_years + abs_contrast + n_trials'
    return formula_full_str, formula_reduced_age_str, formula_reduced_region_str


def def_BF_formula_region(metric, mean_subtraction=False, log_transform=False):
    if metric in ['fr_delta_modulation', 'ff_quench_modulation']:
        formula_full_str = f'{metric} ~ age_years + n_trials'
        formula_reduced_age_str = f'{metric} ~ n_trials'
    elif metric in ['ff_quench']:
        if mean_subtraction:
            formula_full_str = f'{metric} ~ age_years + n_trials'
            formula_reduced_age_str = f'{metric} ~ n_trials'
        else:
            formula_full_str = f'{metric} ~ age_years + abs_contrast + n_trials'
            formula_reduced_age_str = f'{metric} ~ abs_contrast + n_trials'
    else:
        if mean_subtraction:
            if log_transform:
                formula_full_str = f'log_{metric} ~ age_years + n_trials'
                formula_reduced_age_str = f'log_{metric} ~ n_trials'
            else:
                formula_full_str = f'{metric} ~ age_years + n_trials'
                formula_reduced_age_str = f'{metric} ~ n_trials'
        else:
            if log_transform:
                formula_full_str = f'log_{metric} ~ age_years + abs_contrast + n_trials'
                formula_reduced_age_str = f'log_{metric} ~ abs_contrast + n_trials'
            else:
                formula_full_str = f'{metric} ~ age_years + abs_contrast + n_trials'
                formula_reduced_age_str = f'{metric} ~ abs_contrast + n_trials'
    return formula_full_str, formula_reduced_age_str  


def compute_bayes_factor(df, metric,
                         formula_full_str=None,
                         formula_reduced_age_str=None,
                         formula_reduced_region_str=None):
    # pandas -> R
    with localconverter(default_converter + pandas2ri.converter): 
        r_df = ro.conversion.py2rpy(df)
    ro.globalenv['df_r'] = r_df

    ro.globalenv['formula_full'] = Formula(formula_full_str)
    ro.globalenv['formula_reduced_age'] = Formula(formula_reduced_age_str)
    if formula_reduced_region_str is not None:  
        ro.globalenv['formula_reduced_region'] = Formula(formula_reduced_region_str)
    
    ro.r('''
        library(BayesFactor)
        bf_full <- lmBF(formula_full, data = df_r)
        bf_no_age <- lmBF(formula_reduced_age, data = df_r)
        # posterior 
        chains <- posterior(bf_full, iterations = 10000)
        summary_stats <- as.data.frame(summary(chains)$statistics)
        summary_quants <- as.data.frame(summary(chains)$quantiles)
        bf_age <- bf_full / bf_no_age
        assign("bf_age", bf_age, envir = .GlobalEnv)
        assign("summary_stats", summary_stats, envir = .GlobalEnv)
        assign("summary_quants", summary_quants, envir = .GlobalEnv)
    ''')
    # 
    if formula_reduced_region_str is not None: 
        ro.r('''
            bf_no_region <- lmBF(formula_reduced_region, data = df_r)
            bf_region <- bf_full / bf_no_region
            assign("bf_region", bf_region, envir = .GlobalEnv)
        ''')

    # 
    with localconverter(default_converter + pandas2ri.converter):
        stats_df = ro.conversion.rpy2py(ro.r['summary_stats'])
        quants_df = ro.conversion.rpy2py(ro.r['summary_quants'])
    summary_df = pd.concat([stats_df, quants_df], axis=1)

    
    BF10_age = float(ro.r('extractBF(bf_age)$bf[1]')[0])
    BF10_region = np.nan
    if formula_reduced_region_str is not None:  
        BF10_region = float(ro.r('extractBF(bf_region)$bf[1]')[0])

    return BF10_age, BF10_region, summary_df


def safe_lookup(df, row, col, default=np.nan):
    try:
        return df.loc[row, col]
    except Exception:
        return default


def clean_dependent_variable(df, y_col):
    return df[np.isfinite(df[y_col]) & ~df[y_col].isna()].copy()


def main(mean_subtraction=False, log_transform=True):

    if mean_subtraction:
        metrics_path = C.DATAPATH / "neural_metrics_summary_meansub.parquet"
        selected_metrics = C.METRICS_WITH_MEANSUB
    else:
        metrics_path = C.DATAPATH / "neural_metrics_summary_conditions.parquet"
        selected_metrics = C.METRICS_WITHOUT_MEANSUB

    print("Loading extracted neural metrics summary...")
    neural_metrics = read_table(metrics_path)
    neural_metrics = add_age_group(neural_metrics)

    neural_metrics['log_pre_fr'] = np.log(neural_metrics['pre_fr']+ 1e-6)
    neural_metrics['log_post_fr'] = np.log(neural_metrics['post_fr']+ 1e-6)
    neural_metrics['log_pre_ff'] = np.log(neural_metrics['pre_ff']+ 1e-6)
    neural_metrics['log_post_ff'] = np.log(neural_metrics['post_ff']+ 1e-6)

    result_list = []

    for metric, _ in selected_metrics:
        neural_metrics2use = clean_dependent_variable(neural_metrics, metric)

        formula_full_str, formula_reduced_age_str, formula_reduced_region_str = def_BF_formula(
            metric, mean_subtraction=mean_subtraction, log_transform=log_transform
        )

        if metric in ['fr_delta_modulation', 'ff_quench_modulation']:
            neural_metrics2use = neural_metrics2use.drop(columns=['signed_contrast', 'abs_contrast'], errors='ignore')
            neural_metrics2use = neural_metrics2use.drop_duplicates(
                subset=['uuids', 'cluster_region','session_pid', 'age_years', 'age_group', metric]
            )

        BF10_age, BF10_region, chain_table = compute_bayes_factor(
            neural_metrics2use, metric=metric,
            formula_full_str=formula_full_str,
            formula_reduced_age_str=formula_reduced_age_str,
            formula_reduced_region_str=formula_reduced_region_str
        )

        chain_table.to_csv(C.RESULTSPATH / f'omnibus_{metric}_{get_suffix(mean_subtraction)}BF_chain_table.csv')

        BF10_age_category = interpret_bayes_factor(BF10_age)
        BF10_region_category = interpret_bayes_factor(BF10_region)

        result_df = pd.DataFrame({
            'metric': [metric],
            'formula_full': [formula_full_str],
            'BF10_age': [BF10_age],
            'BF10_age_category': [BF10_age_category],
            'BF10_region': [BF10_region],
            'mean_contrast': [safe_lookup(chain_table, 'abs_contrast-abs_contrast', 'Mean')],
            'low_ci_contrast': [safe_lookup(chain_table, 'abs_contrast-abs_contrast', '2.5%')],
            'high_ci_contrast': [safe_lookup(chain_table, 'abs_contrast-abs_contrast', '97.5%')],
            'mean_age': [safe_lookup(chain_table, 'age_years-age_years', 'Mean')],
            'low_ci_age': [safe_lookup(chain_table, 'age_years-age_years', '2.5%')],
            'high_ci_age': [safe_lookup(chain_table, 'age_years-age_years', '97.5%')]
        })
        result_list.append(result_df)

        # ---- region-specific ----
        result_region_list = []
        for region in C.ROIS:
            print(f"Processing region: {region}")
            formula_full_r, formula_reduced_age_r = def_BF_formula_region(  
                metric, mean_subtraction=mean_subtraction, log_transform=log_transform
            )
            region_df = neural_metrics2use[neural_metrics2use['cluster_region'] == region]
            if len(region_df) > 0:
                BF10_age_region, _, chain_table_region = compute_bayes_factor( 
                    region_df, metric=metric,
                    formula_full_str=formula_full_r,
                    formula_reduced_age_str=formula_reduced_age_r,
                    formula_reduced_region_str=None
                )
                chain_table_region.to_csv(C.RESULTSPATH / f'regional_{metric}_{region}_{get_suffix(mean_subtraction)}BF_chain_table.csv')

                BF10_age_category_region = interpret_bayes_factor(BF10_age_region)

                result_df_region = pd.DataFrame({
                    'cluster_region': [region],
                    'metric': [f"{metric}"],
                    'formula_full': [formula_full_r],
                    'BF10_age': [BF10_age_region],
                    'BF10_age_category': [BF10_age_category_region],
                    'BF10_region': [np.nan],  
                    'mean_contrast': [safe_lookup(chain_table_region, 'abs_contrast-abs_contrast', 'Mean')],
                    'low_ci_contrast': [safe_lookup(chain_table_region, 'abs_contrast-abs_contrast', '2.5%')],
                    'high_ci_contrast': [safe_lookup(chain_table_region, 'abs_contrast-abs_contrast', '97.5%')],
                    'mean_age': [safe_lookup(chain_table_region, 'age_years-age_years', 'Mean')],
                    'low_ci_age': [safe_lookup(chain_table_region, 'age_years-age_years', '2.5%')],
                    'high_ci_age': [safe_lookup(chain_table_region, 'age_years-age_years', '97.5%')]
                })
                result_region_list.append(result_df_region)

        if result_region_list:
            final_df_region = pd.concat(result_region_list, ignore_index=True)
            final_df_region['age_ci_conclusion'] = ~((final_df_region['low_ci_age'] < 0) & (final_df_region['high_ci_age'] > 0))
            if mean_subtraction:
                final_df_region.to_csv(C.RESULTSPATH / f'regional_{get_suffix(mean_subtraction)}BFs_{C.ALIGN_EVENT}_{metric}_{C.TRIAL_TYPE}.csv')
            else:
                final_df_region['contrast_ci_conclusion'] = ~((final_df_region['low_ci_contrast'] < 0) & (final_df_region['high_ci_contrast'] > 0))
                final_df_region.to_csv(C.RESULTSPATH / f'regional_{get_suffix(mean_subtraction)}BFs_{C.ALIGN_EVENT}_{metric}_{C.TRIAL_TYPE}.csv')

    final_df = pd.concat(result_list, ignore_index=True)
    final_df['age_ci_conclusion'] = ~((final_df['low_ci_age'] < 0) & (final_df['high_ci_age'] > 0))
    if not mean_subtraction:
        final_df['contrast_ci_conclusion'] = ~((final_df['low_ci_contrast'] < 0) & (final_df['high_ci_contrast'] > 0))
    filename = C.RESULTSPATH / f'omnibus_{get_suffix(mean_subtraction)}BFs_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}.csv'
    final_df.to_csv(filename, index=False)
    log.info(f"[Saved table] {filename.resolve()}")


if __name__ == "__main__":
    from scripts.utils.io import setup_logging
    setup_logging()

    for mean_sub in (True, False):
        logging.info(f"=== Run with mean_subtraction={mean_sub} ===")
        main(mean_subtraction=mean_sub,
             log_transform=True)
