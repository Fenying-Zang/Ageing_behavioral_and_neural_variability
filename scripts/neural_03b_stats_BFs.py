"""
compute BFs with BayesFactor package, without any random effects structure


"""
#%%
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
from ibl_style.utils import MM_TO_INCH
from ibl_style.style import figure_style
from matplotlib.ticker import MultipleLocator
import figrid as fg

from utils.config import datapath, align_event, trial_type, ROIs, age_group_threshold,metrics_with_meansub,metrics_without_meansub  
from rpy2.robjects import Formula
import math
import os
# Set up R environment
os.environ['R_HOME'] = r"C:/PROGRA~1/R/R-44~1.1"
os.environ['R_USER'] = os.path.expanduser("~")
os.environ["R_DISABLE_CONSOLE_OUTPUT"] = "TRUE"  # 🚫 关闭 R → Python 控制台输出（避免 cffi 问题）


# 设置 R 的 locale 为 UTF-8
import rpy2.robjects as ro

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# Activate pandas ↔ R automatic conversion
pandas2ri.activate()
# Load BayesFactor package
bayesfactor = importr('BayesFactor')

# Disable JIT (avoids rpy2 issues on Windows)
ro.r('compiler::enableJIT(0)')
ro.r('Sys.setlocale("LC_ALL", "English_United States.UTF-8")')

def load_neural_metrics_data(df_path):
    """
    Load extracted metrics summary data
    """
    return pd.read_parquet(df_path)

def get_suffix(mean_subtraction):
    return 'meansub' if mean_subtraction else ''

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
    
def def_BF_formula(metric, mean_subtraction=False, log_transform=False):
    # 定义模型公式
    if metric in [ 'fr_delta_modulation', 'ff_quench_modulation']:
        formula_full_str = f'{metric} ~ age_years + cluster_region + n_trials'
        formula_reduced_age_str = f'{metric} ~ cluster_region + n_trials'
        formula_reduced_region_str = f'{metric} ~ age_years + n_trials'
    elif metric in ['ff_quench']:
        if mean_subtraction:
            formula_full_str = f'{metric} ~ age_years + n_trials + cluster_region'
            formula_reduced_age_str = f'{metric} ~   n_trials + cluster_region'
            formula_reduced_region_str = f'{metric} ~ age_years  + n_trials '
        else:
            formula_full_str = f'{metric} ~ age_years + abs_contrast + n_trials + cluster_region'
            formula_reduced_age_str = f'{metric} ~ abs_contrast + n_trials + cluster_region'
            formula_reduced_region_str = f'{metric} ~ age_years + abs_contrast + n_trials '

    else:
        if mean_subtraction:
            if log_transform:
                formula_full_str = f'log_{metric} ~ age_years  + n_trials + cluster_region'
                formula_reduced_age_str = f'log_{metric} ~ n_trials + cluster_region '
                formula_reduced_region_str = f'log_{metric} ~ age_years  + n_trials '
            else:
                formula_full_str = f'{metric} ~ age_years  + n_trials + cluster_region'
                formula_reduced_age_str = f'{metric} ~ n_trials + cluster_region '
                formula_reduced_region_str = f'{metric} ~ age_years  + n_trials '
        else:
            if log_transform:
                formula_full_str = f'log_{metric} ~ age_years + abs_contrast + n_trials + cluster_region'
                formula_reduced_age_str = f'log_{metric} ~ abs_contrast + n_trials + cluster_region'
                formula_reduced_region_str = f'log_{metric} ~ age_years + abs_contrast + n_trials '
            else:
                formula_full_str = f'{metric} ~ age_years + abs_contrast + n_trials + cluster_region'
                formula_reduced_age_str = f'{metric} ~ abs_contrast + n_trials + cluster_region'
                formula_reduced_region_str = f'{metric} ~ age_years + abs_contrast + n_trials '

    
    return formula_full_str, formula_reduced_age_str, formula_reduced_region_str


def def_BF_formula_region(metric, mean_subtraction=False, log_transform=False):
    # 定义模型公式
    if metric in [ 'fr_delta_modulation', 'ff_quench_modulation']:
        formula_full_str = f'{metric} ~ age_years  + n_trials'
        formula_reduced_age_str = f'{metric} ~  n_trials'
    elif metric in ['ff_quench']:
        if mean_subtraction:
            formula_full_str = f'{metric} ~ age_years  + n_trials '
            formula_reduced_age_str = f'{metric} ~ n_trials '
        else:
            formula_full_str = f'{metric} ~ age_years + abs_contrast + n_trials '
            formula_reduced_age_str = f'{metric} ~ abs_contrast + n_trials '

    else:
        if mean_subtraction:
            if log_transform:
                formula_full_str = f'log_{metric} ~ age_years  + n_trials '
                formula_reduced_age_str = f'log_{metric} ~ n_trials  '
            else:
                formula_full_str = f'{metric} ~ age_years  + n_trials '
                formula_reduced_age_str = f'{metric} ~ n_trials  '
        else:
            if log_transform:
                formula_full_str = f'log_{metric} ~ age_years + abs_contrast + n_trials '
                formula_reduced_age_str = f'log_{metric} ~ abs_contrast + n_trials '
            else:
                formula_full_str = f'{metric} ~ age_years + abs_contrast + n_trials '
                formula_reduced_age_str = f'{metric} ~ abs_contrast + n_trials '
    return formula_full_str, formula_reduced_age_str, formula_reduced_region_str

def compute_bayes_factor(df, metric,
                         formula_full_str= None, formula_reduced_age_str=None, 
                         formula_reduced_region_str=None):

    ro.globalenv['df_r'] = pandas2ri.py2rpy(df)

    ro.globalenv['formula_full'] = Formula(formula_full_str)
    ro.globalenv['formula_reduced_age'] = Formula(formula_reduced_age_str)
    ro.globalenv['formula_reduced_region'] = Formula(formula_reduced_region_str)

    # # Run Bayes Factor analysis in R
    ro.r('''
    library(BayesFactor)

    #df_r$dummy <- 1 # Add a dummy variable for the reduced model
    # Fit models
    bf_full <- lmBF(formula_full, data = df_r)
    bf_no_age <- lmBF(formula_reduced_age, data = df_r)
    bf_no_region <- lmBF(formula_reduced_region, data = df_r)

    ## Sample from the posterior of the full model
    chains = posterior(bf_full, iterations = 10000)
    ## 1:13 are the only "interesting" parameters
    #summary(chains[,1:22])
    summary_stats <- as.data.frame(summary(chains)$statistics)
    summary_quants <- as.data.frame(summary(chains)$quantiles)
         

    # Compute Inclusion Bayes Factor
    bf_age <- bf_full / bf_no_age
    bf_region <- bf_full / bf_no_region
    #bf_numeric <- extractBF(bf_age)$bf    
    extractBF(bf_age)$bf
    extractBF(bf_region)$bf     
         
    assign("bf_age", bf_age, envir = .GlobalEnv)
    assign("bf_region", bf_region, envir = .GlobalEnv)
    ''')

    # 转为 pandas.DataFrame
    stats_df = pandas2ri.rpy2py(ro.r['summary_stats'])
    quants_df = pandas2ri.rpy2py(ro.r['summary_quants'])

    # 合并为一个表（可选）
    summary_df = pd.concat([stats_df, quants_df], axis=1)

    return ro.r('extractBF(bf_age)$bf[1]'),ro.r('extractBF(bf_region)$bf[1]'), summary_df


def safe_lookup(df, row, col, default=np.nan):
    try:
        return df.loc[row, col]
    except KeyError:
        return default

def clean_dependent_variable(df, y_col):
    return df[np.isfinite(df[y_col]) & ~df[y_col].isna()].copy()

if __name__ == "__main__":
    mean_subtraction = False #TODO:
    log_transform = True

    if mean_subtraction:
        metrics_path = datapath / "neural_metrics_summary_meansub_merged.parquet"
        selected_metrics = metrics_with_meansub
    else:
        metrics_path = datapath / "neural_metrics_summary_conditions_merged.parquet"
        selected_metrics = metrics_without_meansub

    print("Loading extracted neural metrics summary...")
    neural_metrics = load_neural_metrics_data(metrics_path)
    neural_metrics['age_group'] = neural_metrics['mouse_age'].map(lambda x: 'old' if x > age_group_threshold else 'young')
    neural_metrics['mouse_age_months'] = neural_metrics['mouse_age'] / 30
    neural_metrics['age_years'] = neural_metrics['mouse_age'] / 365

    # neural_metrics['pre_frs'] = neural_metrics['pre_frs']+ 1e-6
    neural_metrics['log_pre_fr'] = np.log(neural_metrics['pre_fr']+ 1e-6)
    neural_metrics['log_post_fr'] = np.log(neural_metrics['post_fr']+ 1e-6)

    # neural_metrics['pre_FFs'] = neural_metrics['pre_FFs'] + 1e-6
    neural_metrics['log_pre_ff'] = np.log(neural_metrics['pre_ff']+ 1e-6)
    neural_metrics['log_post_ff'] = np.log(neural_metrics['post_ff']+ 1e-6)

    result_list = []
    for metric, est in selected_metrics:
        neural_metrics2use = clean_dependent_variable(neural_metrics, metric)

        # log transform or not?
        formula_full_str, formula_reduced_age_str, formula_reduced_region_str = def_BF_formula(metric, mean_subtraction=mean_subtraction, log_transform=log_transform)
        if metric in ['fr_delta_modulation', 'ff_quench_modulation']:
            neural_metrics2use = neural_metrics2use.drop(columns=['signed_contrast', 'abs_contrast'], errors='ignore')
            neural_metrics2use = neural_metrics2use.drop_duplicates(subset=['uuids', 'cluster_region','session_pid', 'age_years', 'age_group', metric])

        
        BF10_age, BF10_region, chain_table = compute_bayes_factor(neural_metrics2use, metric=metric,
                                        formula_full_str=formula_full_str, 
                                        formula_reduced_age_str=formula_reduced_age_str,
                                        formula_reduced_region_str=formula_reduced_region_str)
        # if mean_subtraction:
        chain_table.to_csv(os.path.join(datapath, f'omnibus_{metric}_{get_suffix(mean_subtraction)}BF_chain_table.csv'), index=True)
        # else:
            # chain_table.to_csv(os.path.join(datapath, f'omnibus_{metric}_BF_chain_table.csv'), index=True)

        BF10_age_category = interpret_bayes_factor(BF10_age)
        BF10_region_category = interpret_bayes_factor(BF10_region)
        print(metric)
        print(f"Bayes Factor for including age_years: {BF10_age_category}")
        print(f"Bayes Factor for including region: {BF10_region_category}")
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
        result_region_list = []
        for region in ROIs:
            print(f"Processing region: {region}")
            formula_full_str, formula_reduced_age_str, formula_reduced_region_str = def_BF_formula_region(metric, mean_subtraction=mean_subtraction, log_transform=log_transform)
            region_df = neural_metrics2use[neural_metrics2use['cluster_region'] == region]
            if len(region_df) > 0:
                BF10_age_region, BF10_region_region, chain_table_region = compute_bayes_factor(region_df, metric=metric,
                                        formula_full_str=formula_full_str, 
                                        formula_reduced_age_str=formula_reduced_age_str
                                        ,formula_reduced_region_str=formula_reduced_region_str)
                # if mean_subtraction:
                chain_table_region.to_csv(os.path.join(datapath, f'omnibus_{metric}_{region}_{get_suffix(mean_subtraction)}BF_chain_table.csv'), index=True)
                # else:
                    # chain_table_region.to_csv(os.path.join(datapath, f'omnibus_{metric}_{region}_BF_chain_table.csv'), index=True)

                BF10_age_category_region = interpret_bayes_factor(BF10_age_region)
                BF10_region_category_region = interpret_bayes_factor(BF10_region_region)

                result_df_region = pd.DataFrame({
                    'cluster_region': [region],
                    'metric': [f"{metric}"],
                    'formula_full': [formula_full_str],
                    'BF10_age': [BF10_age_region],
                    'BF10_age_category': [BF10_age_category_region],
                    'BF10_region': [BF10_region_category_region],
                    'mean_contrast': [safe_lookup(chain_table_region, 'abs_contrast-abs_contrast', 'Mean')],
                    'low_ci_contrast': [safe_lookup(chain_table_region, 'abs_contrast-abs_contrast', '2.5%')],
                    'high_ci_contrast': [safe_lookup(chain_table_region, 'abs_contrast-abs_contrast', '97.5%')],
                    'mean_age': [safe_lookup(chain_table_region, 'age_years-age_years', 'Mean')],
                    'low_ci_age': [safe_lookup(chain_table_region, 'age_years-age_years', '2.5%')],
                    'high_ci_age': [safe_lookup(chain_table_region, 'age_years-age_years', '97.5%')]
                })
                result_region_list.append(result_df_region)
        final_df_region = pd.concat(result_region_list, ignore_index=True)
        final_df_region['age_ci_conclusion'] = ~((final_df_region['low_ci_age'] < 0) & (final_df_region['high_ci_age'] > 0))
        if mean_subtraction:
            final_df_region.to_csv(os.path.join(datapath, f'regional_{get_suffix(mean_subtraction)}BFs_{align_event}_{metric}_{trial_type}_2025_06Aug.csv'), index=False)
        else:
            final_df_region['contrast_ci_conclusion'] = ~((final_df_region['low_ci_contrast'] < 0) & (final_df_region['high_ci_contrast'] > 0))
            final_df_region.to_csv(os.path.join(datapath, f'regional_{get_suffix(mean_subtraction)}BFs_{align_event}_{metric}_{trial_type}_2025_06Aug_log.csv'), index=False)

    final_df = pd.concat(result_list, ignore_index=True)
    print(final_df)

    final_df['age_ci_conclusion'] = ~((final_df['low_ci_age'] < 0) & (final_df['high_ci_age'] > 0))

    if mean_subtraction:
        final_df.to_csv(os.path.join(datapath, f'Omnibus_{get_suffix(mean_subtraction)}BFs_{align_event}_{trial_type}_2025_06Aug.csv'), index=False)
    else:
        final_df['contrast_ci_conclusion'] = ~((final_df['low_ci_contrast'] < 0) & (final_df['high_ci_contrast'] > 0))
        final_df.to_csv(os.path.join(datapath, f'Omnibus_BFs_{align_event}_{trial_type}_2025_06Aug_log.csv'), index=False)

# %%
