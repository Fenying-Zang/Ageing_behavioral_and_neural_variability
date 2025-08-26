"""
Figure 1 S4. RT variability measures across two RT definitions and their relationship with age.

Plots age effects on RT variability metrics (MAD, CV, SD log RT) for:
- response_times_from_stim
- firstMovement_times_from_stim

Saves: C.FIGPATH / "F1_supp_rt_variations.pdf"
"""
#%%
# =====================
# Imports
# =====================
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.genmod.families import Gaussian
from scripts.utils.plot_utils import figure_style, format_bf_annotation
from ibl_style.utils import get_coords, MM_TO_INCH, double_column_fig
import figrid as fg

from scripts.utils.behavior_utils import filter_trials
from scripts.utils.data_utils import bf_gaussian_via_pearson, interpret_bayes_factor
import config as C 
from scripts.utils.stats_utils import get_permut_results_table
from scripts.utils.io import read_table, save_figure, setup_logging
import logging

log = logging.getLogger(__name__)
# =====================
# Tunables (script-local)
# =====================

MEASURES = ['mad_rt', 'cv_rt', 'sd_log_rt']
Y_LABELS = {'mad_rt': 'MAD of RT', 'cv_rt': 'CV of RT', 'sd_log_rt': 'SD of log RT'}
FAMILY_FUNC = Gaussian()
N_JOBS = 6
SHUFFLING = 'labels1_global'  # same as your other scripts

# =====================
# 2) Prepare data (filter + variability summary)
# =====================
def filter_for_rt(trials, rt_variable_name, exclude_nan_event_trials=True, clean_rt=True):
    """Standard trial filtering for a given RT variable; adds age fields and log_rt; returns a copy."""
    df = filter_trials(
        trials,
        exclude_nan_event_trials=exclude_nan_event_trials,
        trial_type=C.TRIAL_TYPE,
        event_list=C.EVENT_LIST,
        clean_rt=clean_rt,
        rt_variable=rt_variable_name,
        rt_cutoff=C.RT_CUTOFF,
    ).copy()
    df['age_group'] = (df['mouse_age'] > C.AGE_GROUP_THRESHOLD).map({True: "old", False: "young"})
    df['age_months'] = df['mouse_age'] / 30
    df['age_years'] = df['mouse_age'] / 365
    # RT columns
    df = df.dropna(subset=['rt'])
    df['log_rt'] = np.log(df['rt'])
    return df


def compute_subject_measures(group):
    """Per-session variability features."""
    mean_rt = group['rt'].mean()
    sd_rt = group['rt'].std()
    cv_rt = sd_rt / mean_rt if mean_rt not in (0, np.nan) else np.nan
    mad_rt = np.median(np.abs(group['rt'] - np.median(group['rt'])))
    mean_log_rt = group['log_rt'].mean()
    sd_log_rt = group['log_rt'].std()
    cv_log_rt = sd_log_rt / abs(mean_log_rt) if mean_log_rt not in (0, np.nan) else np.nan
    return pd.Series({
        'mean_rt': mean_rt,
        'cv_rt': cv_rt,
        'mad_rt': mad_rt,
        'mean_log_rt': mean_log_rt,
        'sd_log_rt': sd_log_rt,
        'cv_log_rt': cv_log_rt,
        'n_trials': len(group)
    })


def make_variability_summary(filtered_df):
    """Per-session variability metrics (MAD/CV/SD of RT/log RT); one row per eid."""
    keys = ['eid', 'mouse_age', 'age_months', 'age_years', 'age_group']
    out = filtered_df.groupby(keys, as_index=False).apply(compute_subject_measures).reset_index(drop=True)
    print(f"{len(out)} sessions with variability measures")
    return out

# =====================
# 3) Plotting
# =====================
def build_figure_layout():
    figure_style()
    fig = double_column_fig()
    width, height = fig.get_size_inches() / MM_TO_INCH
    xspans = get_coords(width, ratios=[1, 1, 1], space=20, pad=5, span=(0.05, 0.95))
    yspans = get_coords(height, ratios=[1, 1], space=25, pad=5, span=(0, 0.55))

    axs = {
        # response RT
        'response_mad_rt':    fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0]),
        'response_cv_rt':     fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[0]),
        'response_sd_log_rt': fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[0]),
        # movement RT
        'move_mad_rt':        fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1]),
        'move_cv_rt':         fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[1]),
        'move_sd_log_rt':     fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[1]),
    }
    return fig, axs


def plot_rt_variation_panel(ax, variability_summary, measure, perm_df):
    """Scatter/regression + BF/perm annotation for one variability metric within an RT definition."""

    beta = perm_df.loc[perm_df['y_var'] == measure, 'observed_val']
    p_perm = perm_df.loc[perm_df['y_var'] == measure, 'p_perm']
    beta = float(beta.iloc[0]) if len(beta) else np.nan
    p_perm = float(p_perm.iloc[0]) if len(p_perm) else np.nan

    # Bayes factor using Pearson helper
    BF = bf_gaussian_via_pearson(variability_summary, measure, 'age_months')
    BF10 = BF['BF10']
    BF_concl = interpret_bayes_factor(BF10)

    txt = format_bf_annotation(beta, p_perm, BF10, BF_concl, beta_label="age", big_bf=100)

    ax.text(0.05, 1.05, txt, transform=ax.transAxes, fontsize=4, linespacing=0.8, va='top')

    do_reg = (BF_concl in ('strong H1', 'moderate H1'))
    sns.regplot(
        data=variability_summary,
        x='age_months', y=measure,
        fit_reg=do_reg, marker='.', color="1", line_kws=dict(color="gray"), ax=ax
    )
    sns.scatterplot(
        data=variability_summary,
        x='age_months', y=measure,
        hue='age_group', hue_order=['young', 'old'],
        marker='.', legend=False, palette=C.PALETTE, ax=ax
    )

    sns.despine(offset=2, trim=False, ax=ax)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xlabel(None)
    ax.set_ylabel(Y_LABELS.get(measure, measure))


# =====================
# 4) Main
# =====================
def main(save_fig: bool = True):
    trials_table_file = C.DATAPATH / "ibl_included_eids_trials_table2025_full.csv"
    trials = read_table(trials_table_file)

    fig, axs = build_figure_layout()

    # Two RT definitions, plotted into different rows
    for rt_var in ['response_times_from_stim', 'firstMovement_times_from_stim']:
        df_rt = filter_for_rt(trials, rt_var, exclude_nan_event_trials=True, clean_rt=True)
        var_sum = make_variability_summary(df_rt)

        out_csv = C.RESULTSPATH / f"t_2RT_defs_variability_{rt_var}_{C.N_PERMUT_BEHAVIOR}permutation_2025.csv"
        perm_df = get_permut_results_table(
            df=var_sum,
            age_col=C.AGE2USE,                # e.g., 'age_years' or 'age_months'; utils will map to 'age_years'
            measures=MEASURES,
            family_func=FAMILY_FUNC,
            shuffling=SHUFFLING,
            n_permut=C.N_PERMUT_BEHAVIOR,
            n_jobs=N_JOBS,
            random_state=C.RANDOM_STATE,
            filename=out_csv
        )

        # Route to correct row of panels
        row_prefix = 'response' if rt_var == 'response_times_from_stim' else 'move'
        for measure in MEASURES:
            ax = axs[f"{row_prefix}_{measure}"]
            plot_rt_variation_panel(ax, var_sum, measure, perm_df)

    fig.supxlabel('Age (months)', y=0.40)
    fig.supylabel(None)
    plt.tight_layout()

    if save_fig:
        figname = Path(C.FIGPATH) / "F1_supp_rt_variations.pdf"
        save_figure(fig, figname, add_timestamp=True)


if __name__ == "__main__":
    from scripts.utils.io import setup_logging
    setup_logging()

    main(save_fig=True)



























#%%
# #%%
# from scripts.utils.plot_utils import figure_style
# import figrid as fg
# import pickle
# import pandas as pd
# import numpy as np
# import sys, os, time
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from scripts.utils.plot_utils import num_star
# import pingouin as pg
# from scipy import stats
# from scripts.utils.behavior_utils import fit_psychfunc, fit_psychometric_paras
# import scipy as sp
# import seaborn as sns
# import brainbox as bb
# from scripts.utils.behavior_utils import create_trials_table, clean_rts, filter_trials
# from scripts.utils.plot_utils import set_seaborn, plot_psychometric, plot_chronometric
# from scripts.utils.plot_utils import create_slice_org_axes, map_p_value
# import figrid as fg
# from scripts.utils.plot_utils import figure_style
# from ibl_style.utils import MM_TO_INCH
# from one.api import ONE
# from scripts.utils.data_utils import shuffle_labels_perm,bf_gaussian_via_pearson,interpret_bayes_factor
# from ibl_style.utils import get_coords, MM_TO_INCH, double_column_fig

# # from process_behavioral_data import filter_trials #TODO:与clean_rts比较，保留一个！
# from joblib import Parallel, delayed
# from statsmodels.genmod.families import Gaussian
# from statsmodels.formula.api import glm
# from statsmodels.genmod.families import Gamma
# from statsmodels.genmod.families.links import Log 
# from tqdm import tqdm
# from scripts.utils.permutation_test import plot_permut_test
# from scripts.utils.data_utils import shuffle_labels_perm
# import config as C
#                         #   rt_cutoff, event_list, palette, C.AGE_GROUP_THRESHOLD,
#                         #   n_permut_behavior, n_permut_neural_regional, )
# n_permut_behavior =20 #TODO:for testing, set to 10
# save_fig=True
# exclude_nan_event_trials = True
# clean_rt = True
# easy_trials = False

# easy_trials = False # if True, only include trials with signed_contrast >= 50
# permut_result_list= []   
# measures_list = ['mad_rt','cv_rt','sd_log_rt']
# y_labels =  ['MAD of RT','CV of RT','SD of log RT']

# n_jobs =6
# n_permut = 10
# shuffling='labels1_global'#'labels1_based_on_2'
# family_func = Gaussian()
# plot = True
# save_fig=True
# age2use = 'age_years'#'age_years'

# def load_trial_table(filepath):
#     trials_table = pd.read_csv(filepath)
#     print(len(set(trials_table['eid'])), 'sessions loaded')
#     return trials_table

# def get_permut_results (rt_variable_name, measure, age2use, variability_summary):
#     # permut_result_df = pd.read_csv(os.path.join(C.DATAPATH, f'2RT_defs_variability_{n_permut}permutation_2025.csv'))

#     filename = C.DATAPATH / f'2RT_defs_variability_{rt_variable_name},{n_permut_behavior}permutation_2025.csv'
#     if filename.exists():
#         permut_df = pd.read_csv(filename)
#         # p_perm = permut_df['p_perm'].values[0]
#         # observed_val = permut_df['observed_val'].values[0]

#     else:
#         permut_result_list= []   
#         for m, measure in enumerate(measures_list):   
#             # shuffling='labels1_global'#'labels1_based_on_2'
#             family_func = Gaussian()
#             # plot = True
#             formula2use = f"{measure} ~ {age2use} "

#             idxs = ~ np.isnan(variability_summary[measure])
#             region_data = variability_summary[idxs].reset_index(drop=True)

#             this_data = region_data[measure].values
#             this_age = region_data[age2use].values
#             observed_val, observed_val_p, p_perm, valid_null = run_permutation_test(
#                 data=region_data,
#                 this_age=this_age,
#                 formula2use=formula2use,
#                 family_func=family_func,
#                 n_permut=n_permut_behavior,
#                 n_jobs=6
#             )
#             print(f"results for{rt_variable_name}, {measure}: \n  beta = {observed_val:.4f}, p_perm = {p_perm:.4f}")
#             permut_result_list.append({
#                             'rt_variable': rt_variable_name,
#                             'y_var': measure,
#                             'n_perm': n_permut_behavior,
#                             'formula': formula2use,
#                             'observed_val': observed_val,
#                             'observed_val_p': observed_val_p,
#                             'p_perm': p_perm,
#                             'ave_null_dist': valid_null.mean()
#                             # 'null_dist': null_dist
#                             })
#         permut_df = pd.DataFrame(permut_result_list)
#         permut_df.to_csv(filename, index=False) 
#     return permut_df

# def single_permutation(i, data, permuted_label, formula2use, family_func=Gamma(link=Log())):
#     try:
#         shuffled_data = data.copy()
#         shuffled_data[age2use] = permuted_label

#         model = glm(formula=formula2use, data=shuffled_data, family=family_func).fit()
        
#         return model.params[age2use]
#     except Exception as e:
#         print(f"Permutation {i} failed: {e}")
#         return np.nan
    


# def run_permutation_test(data, this_age, formula2use, family_func, n_permut, n_jobs):

#     permuted_labels, _ = shuffle_labels_perm(
#         labels1=this_age,
#         labels2=None,
#         shuffling='labels1_global',
#         n_permut=n_permut,
#         random_state=123,
#         n_cores=n_jobs
#     )

#     null_dist = Parallel(n_jobs=n_jobs)(
#         delayed(single_permutation)(i, data, permuted_labels[i], formula2use, family_func)
#         for i in tqdm(range(n_permut))
#     )

#     null_dist = np.array(null_dist)
#     valid_null = null_dist[~np.isnan(null_dist)]

#     model_obs = glm(formula=formula2use, data=data, family=family_func).fit()
#     observed_val = model_obs.params["age_years"]
#     observed_val_p = model_obs.pvalues["age_years"]
#     p_perm = (np.sum(np.abs(valid_null) >= np.abs(observed_val)) + 1) / (len(valid_null) + 1)

#     return observed_val, observed_val_p, p_perm, valid_null




# def setup_fig_axes(fg, MM_TO_INCH, fig=None):

#     if fig is None:
#         fig = double_column_fig()
#     figure_style()
#     # Make a double column figure
#     fig = double_column_fig()
#     # Get the dimensions of the figure in mm
#     # Get the dimensions of the figure in mm
#     width, height = fig.get_size_inches() / MM_TO_INCH  # 180, 170
#     xspans = get_coords(width, ratios=[1, 1, 1], space=20, pad=5, span=(0.05, 0.95))#from 0-1
#     yspans = get_coords(height, ratios=[1, 1], space=25, pad=5, span=(0, 0.55))

#     # dlc_list = [ 'wheel_velocity', 'speed_paw_l', 'speed_paw_r', 'speed_nose_tip','speed_tail_start']
#     # measure_list =['ave','cv'] #vbt
#     # ['mad_rt','cv_rt','sd_log_rt']
#     axs = {'response_mad_rt': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0]),
#         'response_cv_rt': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[0]),
#         'response_sd_log_rt': fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[0]),

#         'move_mad_rt': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1]),
#         'move_cv_rt': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[1]),
#         'move_sd_log_rt': fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[1]),

#     }
#     return fig, axs

# def compute_subject_measures(group):
#     mean_rt = group['rt'].mean()
#     median_rt = group['rt'].median()
#     sd_rt = group['rt'].std()
#     cv_rt = sd_rt / mean_rt if mean_rt != 0 else np.nan

#     mad_rt = np.median(np.abs(group['rt'] - np.median(group['rt'])))

#     mean_log_rt = group['log_rt'].mean()
#     sd_log_rt = group['log_rt'].std()
#     cv_log_rt = sd_log_rt / abs(mean_log_rt) if mean_log_rt != 0 else np.nan

#     return pd.Series({
#         'mean_rt': mean_rt,
#         'median_rt': median_rt,
#         'cv_rt': cv_rt,
#         'mad_rt': mad_rt,
#         'mean_log_rt': mean_log_rt,
#         'sd_log_rt': sd_log_rt,
#         'cv_log_rt': cv_log_rt,
#         'n_trials': len(group)
#     })

# if __name__ == "__main__":

#     trials_table_file = C.DATAPATH / 'ibl_included_eids_trials_table2025_full.csv'
#     trials_table = load_trial_table(trials_table_file)
#     fig,axs = setup_fig_axes(fg, MM_TO_INCH)

#     for rt_variable_name in ['response_times_from_stim', 'firstMovement_times_from_stim']:
#         data_filtered = filter_trials(trials_table, exclude_nan_event_trials=True, 
#                                             C.TRIAL_TYPE='first400', event_list=event_list, clean_rt=clean_rt, 
#                                             rt_variable=rt_variable_name, rt_cutoff=rt_cutoff)
#         data_filtered['age_group'] = (data_filtered['mouse_age'] > C.AGE_GROUP_THRESHOLD).map({True: "old", False: "young"})
#         data_filtered['age_months'] = data_filtered['mouse_age'] / 30  # Convert age to months
#         data_filtered['age_years'] = data_filtered['mouse_age'] / 365  # Convert age to months

#         # Step 1: Filter RTs
#         data_filtered = data_filtered.dropna(subset=['rt'])   # Remove NaNs in RT first
#         data_filtered['log_rt'] = np.log(data_filtered['rt'])

#         variability_summary = data_filtered.groupby(['eid', 'mouse_age','age_months','age_years','age_group']).apply(compute_subject_measures).reset_index()
#         print(len(variability_summary), 'subjects with variability measures')


#         for m, measure in enumerate(measures_list):    
#             formula2use = f"{measure} ~ {age2use} "
#             idxs = ~ np.isnan(variability_summary[measure])
#             region_data = variability_summary[idxs].reset_index(drop=True)

#             if rt_variable_name == 'response_times_from_stim':
#                 ax = axs[f'response_{measure}']
#             elif rt_variable_name == 'firstMovement_times_from_stim':
#                 ax = axs[f'move_{measure}']
            
#             #load permut results
#             permut_result_df = get_permut_results(rt_variable_name, measure, age2use, variability_summary)



#             beta = permut_result_df[(permut_result_df['rt_variable'] == rt_variable_name)&(permut_result_df['y_var'] == measure)]['observed_val'].values[0]

#             p_perm= permut_result_df[(permut_result_df['rt_variable'] == rt_variable_name)&(permut_result_df['y_var'] == measure)]['p_perm'].values[0]
#             # stars_p_perm = num_star_new(p_perm)

#             BF_dict = bf_gaussian_via_pearson(variability_summary, measure, 'age_months')
#             # print(f"BF10 for {content} vs. {age2use}: {BF_dict['BF10']:.3f}, r={BF_dict['r']:.3f}, n={BF_dict['n']}")
#             BF10 = BF_dict['BF10']
#             BF_conclusion = interpret_bayes_factor(BF10)
#             # stars_p_perm = num_star_new(p_perm)
#             mapped_p_value = map_p_value(p_perm)

#             if BF10 > 100:
#                 txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} {mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} > 100, $" + f" {BF_conclusion}"
#             else:
#                 txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} {mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} = {BF10:.3f}, $" + f" {BF_conclusion}"

#             ax.text(0.05, 1.05, txt , transform=ax.transAxes, fontsize=4, linespacing=0.8,
#                         verticalalignment='top')

#             if BF_conclusion == 'strong H1' or BF_conclusion == 'moderate H1':
#                 sns.regplot(data=variability_summary, x='age_months', y=measure, 
#                         marker='.', color="1", line_kws=dict(color="gray"), ax=ax)
#             else:
#                 sns.regplot(data=variability_summary, x='age_months', y=measure, 
#                         fit_reg=False, marker='.', color="1", line_kws=dict(color="gray"), ax=ax) #color = .3
#             sns.scatterplot(x='age_months', y=measure, data=variability_summary, hue='age_group',
#                     marker='.',legend=False, palette=palette, hue_order=['young','old'], ax=ax)  #marker='o',
            
#             sns.despine(offset=2, trim=False, ax=ax)

#             ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
#             ax.set_xlabel(None)
#             ax.set_ylabel(y_labels[m])  

#     fig.supxlabel('Age (months)', font="Arial",fontsize=7,y=0.4)#, font="Arial",fontsize=22
#     fig.supylabel(None)
#     plt.tight_layout()

#     if save_fig:
#         # fig.savefig(os.path.join(config.C.FIGPATH, "f1_correlation_behavioral_perf_paras_rt_actualage_2025.pdf")) 
#         fig.savefig(os.path.join(C.FIGPATH, f"F1_supp_rt_variations.pdf")) 
