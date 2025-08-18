"""
[kept only the relevant code]
Fenying Zang, 30 June 2025

Fig1: 
- b.histplot: age distribution
- c.psychometric function+ scatter plot of 3 measures
- d.chronometric function (MEDIAN RT) + scatterplot: Median RT with age
- e.chronometric function (RT variabiltiy) 
- f.scatterplot: RT variability with age
- g.histplot: raw RT distribution in both groups
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy.stats import variation

# === IBL / Brainbox Libraries ===
from one.api import ONE
import brainbox as bb
from ibl_style.style import figure_style

# === Custom Project Utilities ===
import config as C
from scripts.utils.plot_utils import (
    plot_psychometric,
    plot_chronometric,
    map_p_value,
)
from scripts.utils.behavior_utils import (
    create_trials_table,
    clean_rts,
    fit_psychfunc,
    fit_psychometric_paras,
    compute_choice_history
)
from scripts.utils.behavior_utils import filter_trials
from scripts.utils.io import read_table, save_figure
# === Plotting Tools ===
import matplotlib.ticker as mticker
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'

# === Configuration ===
one = ONE()
figure_style()
# set_seaborn()

def plot_age_distribution(trials_table,  save_fig=True, session_based=False):
    trials_table['age_months'] = trials_table['mouse_age'] / 30
    if session_based:
        age_info = trials_table.groupby(['age_group', 'eid'])[['age_months']].mean().reset_index()
        fig = sns.displot(
            age_info, x="age_months", hue="age_group",
            hue_order=['young', 'old'], binwidth=0.5,
            palette=C.PALETTE, legend=False, height=2.36, aspect=1,
            multiple='stack'
        )
        fig.set_axis_labels('Age at recording (months)', 'Number of sessions')
        for ax in fig.axes.flat:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        if save_fig:
            save_figure(fig, C.FIGPATH / "f1_distribution_age_allmice_sessionbased_2025.pdf")
        plt.show()
    else:
        age_info = trials_table.groupby(['age_group', 'mouse_name'])[['age_months']].mean().reset_index()
        fig = sns.displot(
            age_info, x="age_months", hue="age_group",
            hue_order=['young', 'old'], binwidth=0.5,
            palette=C.PALETTE, legend=False, height=2.36, aspect=1,
            multiple='stack'
        )
        fig.set_axis_labels('Age (months)', 'Number of mice')

        txt_young = f"Young, " + fr" $n_{{\mathrm{{mice}}}} = {'97'} $"+  f"\nM(age)= {5.50}" 
        txt_old = f"Old, " +fr" $n_{{\mathrm{{mice}}}} ={'52'} $"+  f"\nM(age) = {11.56}" 


        for ax in fig.axes.flat:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.set_yticks(np.arange(0, 21, 4))
            ax.text(0.6, 0.4, txt_old , transform=ax.transAxes, fontsize=7, linespacing=0.8, c=C.PALETTE['old'], 
                verticalalignment='top')

            ax.text(0.4, 0.8, txt_young , transform=ax.transAxes, fontsize=7, linespacing=0.8,  c=C.PALETTE['young'], 
                verticalalignment='top')
        if save_fig:
            save_figure(fig, C.FIGPATH / "f1b_distribution_age_allmice_mousebased_2025.pdf")
        plt.show()


def plot_rt_distribution(trials_table, clean_rt=False, save_fig=True, easy_trials=False):
    trials = filter_trials(
        trials_table, exclude_nan_event_trials=True,
        trial_type=C.TRIAL_TYPE, event_list=C.EVENT_LIST,
        clean_rt= False, rt_variable=C.RT_VARIABLE_NAME,
        rt_cutoff=C.RT_CUTOFF
    )
    print(len(trials))
    # rt_variable=C.RT_VARIABLE_NAME
    print(trials.rt.min())
    trials['age_group'] = (trials['mouse_age'] > C.AGE_GROUP_THRESHOLD).map({True: "old", False: "young"})

    data2plot = trials.loc[trials['signed_contrast'].abs() >= 50] if easy_trials else trials

    bin_labels = ['< %dms' % int(C.RT_CUTOFF[0]*1000),
                  '%dms - %ds' % (int(C.RT_CUTOFF[0]*1000), int(C.RT_CUTOFF[1])),
                  '> %ds' % int(C.RT_CUTOFF[1])]

    data2plot['rt_raw_category'] = pd.cut(
        data2plot['rt_raw'],
        bins=[data2plot.rt_raw.min(), C.RT_CUTOFF[0], C.RT_CUTOFF[1], data2plot.rt_raw.max()],
        labels=bin_labels, right=True
    )
    data2plot.loc[data2plot.rt_raw > C.RT_CUTOFF[1], 'rt_raw'] = C.RT_CUTOFF[1]+0.08
    # data2plot.loc[data2plot.rt_raw < rt_cutoff[0], 'rt_raw'] = 0

    percentage_info = data2plot.groupby(['age_group', 'rt_raw_category'], as_index=False)['trial_index'].count()
    percentage_info['percentage'] = (
        percentage_info['trial_index'] /
        percentage_info['age_group'].map(data2plot.groupby('age_group')['trial_index'].count())
    ) * 100
    percentage_info['percentage'] = percentage_info['percentage'].round(2)
    percentage_info = percentage_info[percentage_info['rt_raw_category'] == '> %ds' % int(C.RT_CUTOFF[1])]

    fig = sns.FacetGrid(
        data=data2plot, hue='age_group', hue_order=['young', 'old'],
        palette=C.PALETTE, height=2.36, aspect=1
    )
    fig.map(sns.histplot, 'rt_raw', legend=True, binwidth=0.08, stat='probability', common_norm=False, alpha=0.5)

    for axidx, ax in enumerate(fig.axes.flat):
        xlabel = 'Response time (s)' if C.RT_VARIABLE_NAME == 'response_times_from_stim' else 'Movement initiation time (s)'
        ax.set(xlabel=xlabel, xlim=[-0.1, C.RT_CUTOFF[1] + 0.05])
        if axidx == 0:
            young_pct = percentage_info.loc[percentage_info['age_group'] == 'young', 'percentage'].values
            old_pct = percentage_info.loc[percentage_info['age_group'] == 'old', 'percentage'].values

            if len(young_pct) > 0:
                ax.text(.80, .5, f"{young_pct[0]:.1f}%", fontsize=6, color=C.PALETTE['young'], transform=ax.transAxes)
            if len(old_pct) > 0:
                ax.text(.80, .4, f"{old_pct[0]:.1f}%", fontsize=6, color=C.PALETTE['old'], transform=ax.transAxes)

            # ax.text(.80, .5, f"{percentage_info['percentage'].loc['young']:.1f}", fontsize=6, color=palette['young'], transform=ax.transAxes)
            # ax.text(.80, .4, f"{percentage_info['percentage'].loc['old']:.1f}", fontsize=6, color=palette['old'], transform=ax.transAxes)
        ax.axvline(x=C.RT_CUTOFF[1],  lw=0.5, ls='--', alpha=0.8, c='gray')
        ax.axvline(x=C.RT_CUTOFF[0],  lw=0.5, ls='--', alpha=0.8, c='gray')

    sns.despine(trim=True)
    if save_fig:
        fname = f"f1_distribution_{'easy_trials_' if easy_trials else ''}rt_raw_2groups_2025.pdf"
        save_figure(fig, C.FIGPATH / fname)
    plt.show()


def plot_psychometric_curves(trials_table, save_fig=True):
    """Plot psychometric functions for each age group."""
    data2fit = filter_trials(
        trials_table, exclude_nan_event_trials=True,
        trial_type=C.TRIAL_TYPE, event_list=C.EVENT_LIST,
        clean_rt=True, rt_variable=C.RT_VARIABLE_NAME,
        rt_cutoff=C.RT_CUTOFF
    )
    data2fit['age_group'] = (data2fit['mouse_age'] > C.AGE_GROUP_THRESHOLD).map({True: "old", False: "young"})

    fig = sns.FacetGrid(data2fit, hue="age_group", palette=C.PALETTE, height=2.36, aspect=1)
    fig.map(plot_psychometric, "signed_contrast", "response", "eid", linewidth=2)
    fig.despine(trim=True)
    fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')

    num_sessions = data2fit.groupby('age_group')['eid'].nunique().to_dict()
    txt_young = f"Young, " + fr" $n_{{\mathrm{{sessions}}}} = {num_sessions['young']} $"
    txt_old = f"Old, " +fr" $n_{{\mathrm{{sessions}}}} ={num_sessions['old']} $"

    for ax in fig.axes.flat:
        ax.text(0.02, 0.95, txt_young, transform=ax.transAxes, fontsize=7, linespacing=0.8,  c=C.PALETTE['young'], 
                verticalalignment='top')
        ax.text(0.02, 0.88, txt_old, transform=ax.transAxes, fontsize=7, linespacing=0.8, c=C.PALETTE['old'], 
                verticalalignment='top')
    if save_fig:
        save_figure(fig, C.FIGPATH / "f1_psychfuncs_allmice_first400trials_cleantrials2025.pdf")


def plot_chronometric_curves(trials_table, clean_rt=True, rt_type = 'rt',
                             save_fig=True):
    """Plot psychometric functions for each age group."""
    data2fit = filter_trials(
        trials_table, exclude_nan_event_trials=True,
        trial_type=C.TRIAL_TYPE, event_list=C.EVENT_LIST,
        clean_rt=clean_rt, rt_variable=C.RT_VARIABLE_NAME,
        rt_cutoff=C.RT_CUTOFF
    )
    data2fit['age_group'] = (data2fit['mouse_age'] > C.AGE_GROUP_THRESHOLD).map({True: "old", False: "young"})

    fig = sns.FacetGrid(data2fit, hue="age_group", palette=C.PALETTE, hue_order=['young','old'], height=2.36, aspect=1)
    fig.map(plot_chronometric, "signed_contrast", rt_type, 
        "eid", estimator='median', palette=C.PALETTE , linewidth=2 )
    for axidx, ax in enumerate(fig.axes.flat):
        if rt_type =='rt':
            ax.set(ylim=(0,0.8))
            ax.set_ylabel('Median RT (s)')
        else: 
            ax.set(ylim=(0,1.5))
    if C.RT_VARIABLE_NAME == 'response_times_from_stim':
        fig.set_axis_labels('Signed contrast (%)', 'Median RT (s)')#,fontsize = 10
    elif C.RT_VARIABLE_NAME == 'firstMovement_times_from_stim':
        fig.set_axis_labels('Signed contrast (%)', 'Movement initiation time(s)')#,fontsize = 10

    fig.despine(trim=True)

    if save_fig:
        save_figure(fig, C.FIGPATH / "f1_chronfuncs_allmice_first400trials_{rt_type}2025.pdf")

def plot_chronometric_rt_variability(trials_table, trial_type='first400', clean_rt=True,rt_type = 'rt',
                                     y_var='rt_CV', save_fig=True):
    """Plot psychometric functions for each age group."""
    data2fit = filter_trials(
        trials_table, exclude_nan_event_trials=True,
        trial_type=C.TRIAL_TYPE, event_list=C.EVENT_LIST,
        clean_rt=clean_rt, rt_variable=C.RT_VARIABLE_NAME,
        rt_cutoff=C.RT_CUTOFF
    )
    data2fit['age_group'] = (data2fit['mouse_age'] > C.AGE_GROUP_THRESHOLD).map({True: "old", False: "young"})
    if y_var == 'rt_CV':
        data2fit[y_var] = data2fit.groupby(['eid', 'signed_contrast'])[rt_type].transform(
            lambda x: variation(x.dropna(), ddof=1)
        )

    elif y_var == 'rt_MAD':
        data2fit[y_var] = data2fit.groupby(['eid', 'signed_contrast'])[rt_type].transform(
            lambda x: stats.median_abs_deviation(x, nan_policy='omit')
        )
    data2fit = data2fit.drop_duplicates(subset=['eid', 'signed_contrast',y_var])

    fig = sns.FacetGrid(data2fit, hue="age_group", palette=C.PALETTE, hue_order=['young','old'],height=2.36, aspect=1)
    fig.map(plot_chronometric, "signed_contrast", y_var, "eid", estimator='mean', palette=C.PALETTE , linewidth=2)
    for axidx, ax in enumerate(fig.axes.flat):
 
        if rt_type == 'rt':
            ax.set(ylim=(0,0.8))
        else: 
            ax.set(ylim=(0,0.4))
        ax.set_ylabel('RT variability (CV)')
    fig.set_axis_labels('Signed contrast (%)', f'RT Varibility ({y_var})')

    fig.despine(trim=True)
    if save_fig:
        save_figure(fig, C.FIGPATH / "f1supp_chronfuncs_{y_var}_first400trials_{rt_type}2025.pdf")

      
def plot_psycho_paras_scatter(fit_psy_paras_age_info, permut_result_df, save_fig=True):
    measures_list = ['abs_bias', 'threshold', 'mean_lapse']
    y_labels = ['Abs bias', 'Threshold', 'Mean lapse']

    fig, axs = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(1.18, 2.36))

    for m, measure in enumerate(measures_list):  
        ax = axs[m]  

        beta = permut_result_df[permut_result_df['y_var'] == measure]['observed_val'].values[0]    
        p_perm= permut_result_df[permut_result_df['y_var'] == measure]['p_perm'].values[0]  
        BF10= permut_result_df[permut_result_df['y_var'] == measure]['BF10'].values[0]
        BF_conclusion = permut_result_df[permut_result_df['y_var'] == measure]['BF_conclusion'].values[0]
        
        mapped_p_value = map_p_value(p_perm)
        if BF10 > 100:
            txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} {mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} > 100, $" + f" {BF_conclusion}"
        else:
            txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} {mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} = {BF10:.3f}, $" + f" {BF_conclusion}"
        
        ax.text(0, 1, txt, transform=ax.transAxes, fontsize=3, linespacing=0.8, verticalalignment='top')

        if BF_conclusion == 'strong H1':
            sns.regplot(data=fit_psy_paras_age_info, x='age_months', y=measure, 
                    marker='.', color="1", line_kws=dict(color="gray"), ax=ax)
        else:
            sns.regplot(data=fit_psy_paras_age_info, x='age_months', y=measure, 
                    fit_reg=False, marker='.', color="1", line_kws=dict(color="gray"), ax=ax)
        sns.scatterplot(x='age_months', y=measure, data=fit_psy_paras_age_info, hue='age_group',
                marker='.', legend=False, palette=C.PALETTE, hue_order=['young','old'], ax=ax)
        
        sns.despine(offset=2, trim=False, ax=ax)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax.set_xlabel(None)
        ax.set_ylabel(y_labels[m])  

    fig.supxlabel('Age (months)', font="Arial", fontsize=7)
    fig.supylabel(None)

    plt.tight_layout()
    if save_fig:
        save_figure(fig, C.FIGPATH / "f1_correlation_behavioral_perf_paras_psychometric_paras_actualage_2025.pdf")

def plot_shift_paras_scatter(fit_psy_paras_age_info, permut_result_df,
                             split_type='block', save_fig=True):
    measures_list = ['bias_shift', 'lapselow_shift', 'lapsehigh_shift'] 
    y_labels = ['Bias shift', 'Lapselow shift', 'Lapsehigh shift']
    fig, axs =plt.subplots(1,3,sharex=True, sharey=False, figsize=(7.08, 2.36))

    for m, measure in enumerate(measures_list):  
        ax = axs[m]  
        beta = permut_result_df[permut_result_df['y_var'] == measure]['observed_val'].values[0]    
        p_perm = permut_result_df[permut_result_df['y_var'] == measure]['p_perm'].values[0]  
        BF10 = permut_result_df[permut_result_df['y_var'] == measure]['BF10'].values[0]
        BF_conclusion = permut_result_df[permut_result_df['y_var'] == measure]['BF_conclusion'].values[0]
        
        mapped_p_value = map_p_value(p_perm)
        if BF10 > 100:
            txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}}{mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} > 100, $" + f" {BF_conclusion}"
        else:
            txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}}{mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} = {BF10:.3f}, $" + f" {BF_conclusion}"
        
        ax.text(0.05, 1, txt, transform=ax.transAxes, fontsize=4,  
                verticalalignment='top')

        if BF_conclusion == 'strong H1':
            sns.regplot(data=fit_psy_paras_age_info, x='age_months', y=measure, 
                    marker='.', color="1", line_kws=dict(color="gray"), ax=ax)
        else:
            sns.regplot(data=fit_psy_paras_age_info, x='age_months', y=measure, 
                    fit_reg=False, marker='.', color="1", line_kws=dict(color="gray"), ax=ax)
        sns.scatterplot(x='age_months', y=measure, data=fit_psy_paras_age_info, hue='age_group',
                marker='.', legend=False, palette=C.PALETTE, hue_order=['young','old'], ax=ax) 
        
        sns.despine(offset=2, trim=False, ax=ax)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax.set_xlabel(None)
        ax.set_ylabel(y_labels[m])  

    fig.supxlabel('Age (months)', font="Arial",fontsize=7)
    fig.supylabel(None)

    plt.tight_layout()
    if save_fig:
        save_figure(fig, C.FIGPATH / "f1_correlation_{split_type}_bias_behavioral_perf_paras_psychometric_paras_actualage_2025.pdf")

def plot_rt_paras_scatter(fit_psy_paras_age_info,permut_result_df,save_fig=True):
    measures_list = ['rt_median','rt_CV'] #TODO:rt_MAD,'corr_rt_median','corr_rt_CV','corr_rt_MAD'
    y_labels = ['Median RT (s)','RT variability (CV)']
    for m, measure in enumerate(measures_list):  
        print(f"Processing measure: {measure}")
        try:  
            if measure == 'rt_CV':
                fig, ax =plt.subplots(1,1,sharex=True, sharey=False, figsize=(2.36, 2.36)) #2.36 for large; 

            else:
                fig, ax =plt.subplots(1,1,sharex=True, sharey=False, figsize=(1.18, 1.18)) #2.36 for large; 

            beta = permut_result_df[permut_result_df['y_var'] == measure]['observed_val'].values[0]    
            p_perm= permut_result_df[permut_result_df['y_var'] == measure]['p_perm'].values[0]  
            BF10= permut_result_df[permut_result_df['y_var'] == measure]['BF10'].values[0]
            BF_conclusion = permut_result_df[permut_result_df['y_var'] == measure]['BF_conclusion'].values[0]
            
            mapped_p_value = map_p_value(p_perm)
            if BF10 > 100:
                txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}}{mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} > 100, $" + f" {BF_conclusion}"
            else:
                txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}}{mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} = {BF10:.3f}, $" + f" {BF_conclusion}"
            
            if measure == 'rt_CV':
                ax.text(0.05,  1.05, txt , transform=ax.transAxes, fontsize=6,   
                        verticalalignment='top')
            else:
                ax.text(0, 1.05, txt , transform=ax.transAxes, fontsize=3, linespacing=0.8,  
                        verticalalignment='top')

            if BF_conclusion == 'strong H1' :
                sns.regplot(data=fit_psy_paras_age_info, x='age_months', y=measure, 
                        marker='.', color="1", line_kws=dict(color="gray"), ax=ax)
            else:
                sns.regplot(data=fit_psy_paras_age_info, x='age_months', y=measure, 
                        fit_reg=False, marker='.', color="1", line_kws=dict(color="gray"), ax=ax) #color = .3
            sns.scatterplot(x='age_months', y=measure, data=fit_psy_paras_age_info, hue='age_group',
                    marker='.', legend=False, palette=C.PALETTE, hue_order=['young','old'], ax=ax)  #marker='o',
            
            sns.despine(offset=2, trim=False, ax=ax)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
            ax.set_xlabel(None)
            ax.set_ylabel(y_labels[m])  

            fig.supxlabel('Age (months)', font="Arial", fontsize=7)
            fig.supylabel(None)

            plt.tight_layout()
            if save_fig:
                save_figure(fig, C.FIGPATH / "f1_correlation_behavioral_perf_paras_{measure}_actualage_2025.pdf")

        except Exception as e:
            print(f"Error with measure {measure}: {e}")
            continue


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


def main():
    trials_table_file = C.DATAPATH / 'ibl_included_eids_trials_table2025_full.csv'
    trials_table = load_trial_table(trials_table_file)
    trials_table['age_group'] = (trials_table['mouse_age'] > C.AGE_GROUP_THRESHOLD).map({True: "old", False: "young"})
    
    if trials_table is not None:
        # - b.histplot: age distribution
        plot_age_distribution(trials_table,  save_fig=True, session_based=False)
        # # - g.histplot: raw RT distribution in both groups
        plot_rt_distribution(trials_table,  clean_rt=False, save_fig=True, easy_trials=False)
        # # - c.psychometric function 
        plot_psychometric_curves(trials_table,  save_fig=True)
        # #TODO:
        # # - d.chronometric function (MEDIAN RT) + scatterplot: Median RT with age
        plot_chronometric_curves(trials_table,  clean_rt=True,rt_type = 'rt',
                             save_fig=True)
        # # - e.chronometric function (RT variabiltiy) 
        plot_chronometric_rt_variability(trials_table,  clean_rt=True,rt_type = 'rt',
                                     y_var = 'rt_CV',save_fig=True)

        #TODO: load fitted results
        split_type = 'prevresp'#'block'
        # eid_unique = trials_table.eid.nunique()
        fit_psy_paras_age_info = read_table(C.RESULTSPATH / f"{split_type}_fit_psy_paras_age_info_367sessions_2025.csv")

        label='psychometric'
        permut_result_df = read_table(C.RESULTSPATH, f'permutation_test_{label}_{C.AGE2USE}_{C.N_PERMUT_BEHAVIOR}perm_2025.csv')

        # - c. scatter plot:  of 3 psychometric function measures
        plot_psycho_paras_scatter(fit_psy_paras_age_info, permut_result_df, save_fig=True)
        label='shift'
        permut_result_df = read_table(C.RESULTSPATH, f'permutation_test_{label}_{C.AGE2USE}_{C.N_PERMUT_BEHAVIOR}perm_2025.csv')

        plot_shift_paras_scatter(fit_psy_paras_age_info, permut_result_df, split_type=split_type, save_fig=True)
        label='rt'
        permut_result_df = read_table(C.RESULTSPATH, f'permutation_test_{label}_{C.AGE2USE}_{C.N_PERMUT_BEHAVIOR}perm_2025.csv')
        
        # - f.scatterplot: RT variability with age
        plot_rt_paras_scatter(fit_psy_paras_age_info, permut_result_df, save_fig=True)


if __name__ == "__main__":
    main()
