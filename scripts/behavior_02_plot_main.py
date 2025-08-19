"""
Fenying Zang, 30 June 2025

Fig1: 
- b.histplot: age distribution
- c.psychometric function+ scatter plot of 3 measures
- d.chronometric function (Median RT) + scatterplot: Median RT with age
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
from scripts.utils.plot_utils import figure_style

# === Custom Project Utilities ===
import config as C
from scripts.utils.plot_utils import (
    plot_psychometric,
    plot_chronometric,
    map_p_value,
    format_bf_annotation
)
from scripts.utils.behavior_utils import (
    create_trials_table,
    clean_rts,
    fit_psychfunc,
    fit_psychometric_paras,
    compute_choice_history
)
from scripts.utils.data_utils import add_age_group, add_age_months, add_age_years

from scripts.utils.behavior_utils import filter_trials
from scripts.utils.io import read_table, save_figure
# === Plotting Tools ===
import matplotlib.ticker as mticker
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'

# === Configuration ===
one = ONE()
figure_style()



# def plot_age_distribution(trials_table,  save_fig=True, session_based=False):
#     """_summary_

#     Args:
#         trials_table (dataframe): all trials included
#         save_fig (bool, optional): Defaults to True.
#         session_based (bool, optional): mice who contributed several recording sessions may have several age info in the database, here you can choose decribe age distribution in which way. Defaults to False.
#     """
#     trials_table['age_months'] = trials_table['mouse_age'] / 30
#     if session_based:
#         age_info = trials_table.groupby(['age_group', 'eid'])[['age_months']].mean().reset_index()
#         fig = sns.displot(
#             age_info, x="age_months", hue="age_group",
#             hue_order=['young', 'old'], binwidth=0.5,
#             palette=C.PALETTE, legend=False, height=2.36, aspect=1,
#             multiple='stack'
#         )
#         fig.set_axis_labels('Age at recording (months)', 'Number of sessions')
#         for ax in fig.axes.flat:
#             ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
#         if save_fig:
#             save_figure(fig, C.FIGPATH / "f1_distribution_age_allmice_sessionbased_2025.pdf")
#         plt.show()
#     else:
#         age_info = trials_table.groupby(['age_group', 'mouse_name'])[['age_months']].mean().reset_index()
#         fig = sns.displot(
#             age_info, x="age_months", hue="age_group",
#             hue_order=['young', 'old'], binwidth=0.5,
#             palette=C.PALETTE, legend=False, height=2.36, aspect=1,
#             multiple='stack'
#         )
#         fig.set_axis_labels('Age (months)', 'Number of mice')

#         txt_young = f"Young, " + fr" $n_{{\mathrm{{mice}}}} = {'97'} $"+  f"\nM(age) = {5.50}" 
#         txt_old = f"Old, " +fr" $n_{{\mathrm{{mice}}}} = {'52'} $"+  f"\nM(age) = {11.56}" 


#         for ax in fig.axes.flat:
#             ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
#             ax.set_yticks(np.arange(0, 21, 4))
#             ax.text(0.6, 0.4, txt_old , transform=ax.transAxes, fontsize=7, linespacing=0.8, c=C.PALETTE['old'], 
#                 verticalalignment='top')

#             ax.text(0.4, 0.8, txt_young , transform=ax.transAxes, fontsize=7, linespacing=0.8,  c=C.PALETTE['young'], 
#                 verticalalignment='top')
#             sns.despine(offset=2, trim=False, ax=ax)
#         if save_fig:
#             save_figure(fig, C.FIGPATH / "f1b_distribution_age_allmice_mousebased_2025.pdf")
#         plt.show()
def plot_age_distribution(trials_table, save_fig=True, session_based=False):
    """Plot age distribution (mouse-based or session-based).

    Non-default settings
    --------------------
    session_based : bool
        If True, count sessions (mean age per eid); else count mice (mean age per mouse).
    """
    df = add_age_months(trials_table)

    if session_based:
        age_info = (df.groupby(["age_group", "eid"], as_index=False)["age_months"]
                      .mean())
        xlab = "Age at recording (months)"
        ylab = "Number of sessions"
    else:
        age_info = (df.groupby(["age_group", "mouse_name"], as_index=False)["age_months"]
                      .mean())
        xlab = "Age (months)"
        ylab = "Number of mice"

    # dynamic annotation stats
    stats_df = (age_info.groupby("age_group")["age_months"]
                        .agg(n="count", mean="mean")
                        .reset_index())
    stats_map = {r["age_group"]: r for _, r in stats_df.iterrows()}

    g = sns.displot(
        age_info, x="age_months", hue="age_group",
        hue_order=["young", "old"], binwidth=0.5,
        palette=C.PALETTE, legend=False, height=2.36, aspect=1,
        multiple="stack"
    )
    g.set_axis_labels(xlab, ylab)

    # annotate per-axes (FacetGrid returns a figure-like obj)
    for ax in g.axes.flat:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        # helpful y ticks for dense hist
        try:
            ymax = max(ax.get_yticks()) if len(ax.get_yticks()) else 0
            step = max(1, int(round(ymax / 5))) if ymax else 1
            ax.set_yticks(np.arange(0, ymax + step, step))
        except Exception:
            pass

        # dynamic text blocks
        if "young" in stats_map:
            st = stats_map["young"]
            txt = f"Young, " + r"$n_{\mathrm{mice}}=" + f"{st['n']}" + r"$" + f"\nM(age)={st['mean']:.2f}"
            ax.text(0.40, 0.80, txt, transform=ax.transAxes, fontsize=7,
                    linespacing=0.8, color=C.PALETTE["young"], va="top")
        if "old" in stats_map:
            st = stats_map["old"]
            txt = f"Old, " + r"$n_{\mathrm{mice}}=" + f"{st['n']}" + r"$" + f"\nM(age)={st['mean']:.2f}"
            ax.text(0.60, 0.40, txt, transform=ax.transAxes, fontsize=7,
                    linespacing=0.8, color=C.PALETTE["old"], va="top")

        sns.despine(offset=2, trim=False, ax=ax)

    if save_fig:
        fname = "f1b_distribution_age_allmice_sessionbased_2025.pdf" if session_based \
                else "f1b_distribution_age_allmice_mousebased_2025.pdf"
        save_figure(g, C.FIGPATH / fname)

    plt.show()


# def plot_rt_distribution(trials_table, clean_rt=False, save_fig=True, easy_trials=False):
#     trials = filter_trials(
#         trials_table, exclude_nan_event_trials=True,
#         trial_type=C.TRIAL_TYPE, event_list=C.EVENT_LIST,
#         clean_rt= False, rt_variable=C.RT_VARIABLE_NAME,
#         rt_cutoff=C.RT_CUTOFF
#     )
#     print(len(trials))
#     print(trials.rt.min())
#     trials['age_group'] = (trials['mouse_age'] > C.AGE_GROUP_THRESHOLD).map({True: "old", False: "young"})

#     data2plot = trials.loc[trials['signed_contrast'].abs() >= 50] if easy_trials else trials

#     bin_labels = ['< %dms' % int(C.RT_CUTOFF[0]*1000),
#                   '%dms - %ds' % (int(C.RT_CUTOFF[0]*1000), int(C.RT_CUTOFF[1])),
#                   '> %ds' % int(C.RT_CUTOFF[1])]

#     data2plot['rt_raw_category'] = pd.cut(
#         data2plot['rt_raw'],
#         bins=[data2plot.rt_raw.min(), C.RT_CUTOFF[0], C.RT_CUTOFF[1], data2plot.rt_raw.max()],
#         labels=bin_labels, right=True
#     )
#     data2plot.loc[data2plot.rt_raw > C.RT_CUTOFF[1], 'rt_raw'] = C.RT_CUTOFF[1]+0.08
#     # data2plot.loc[data2plot.rt_raw < rt_cutoff[0], 'rt_raw'] = 0

#     percentage_info = data2plot.groupby(['age_group', 'rt_raw_category'], as_index=False)['trial_index'].count()
#     percentage_info['percentage'] = (
#         percentage_info['trial_index'] /
#         percentage_info['age_group'].map(data2plot.groupby('age_group')['trial_index'].count())
#     ) * 100
#     percentage_info['percentage'] = percentage_info['percentage'].round(2)
#     percentage_info = percentage_info[percentage_info['rt_raw_category'] == '> %ds' % int(C.RT_CUTOFF[1])]

#     # fig = sns.FacetGrid(
#     #     data=data2plot, hue='age_group', hue_order=['young', 'old'],
#     #     palette=C.PALETTE, height=2.36, aspect=1
#     # )
#     # fig.map(sns.histplot, 'rt_raw', legend=True, binwidth=0.08, stat='probability', 
#     #         multiple='dodge', common_norm=False) #, alpha=0.5
    
#     fig = sns.FacetGrid(data=data2plot, height=2.36, aspect=1)
#     fig.map_dataframe(
#         sns.histplot,
#         x='rt_raw',
#         hue='age_group',              
#         binwidth=0.08,
#         stat='probability',
#         multiple='dodge',
#         common_norm=False,
#         palette=C.PALETTE,
#         shrink=0.85                  
#     )
#     fig.add_legend()
#     # sns.despine(trim=True)
#     for axidx, ax in enumerate(fig.axes.flat):
#         xlabel = 'Response time (s)' if C.RT_VARIABLE_NAME == 'response_times_from_stim' else 'Movement initiation time (s)'
#         ax.set(xlabel=xlabel, xlim=[0, C.RT_CUTOFF[1] + 0.05])
#         if axidx == 0:
#             young_pct = percentage_info.loc[percentage_info['age_group'] == 'young', 'percentage'].values
#             old_pct = percentage_info.loc[percentage_info['age_group'] == 'old', 'percentage'].values

#             if len(young_pct) > 0:
#                 ax.text(.80, .5, f"{young_pct[0]:.1f}%", color=C.PALETTE['young'], fontsize=7, transform=ax.transAxes)
#             if len(old_pct) > 0:
#                 ax.text(.80, .4, f"{old_pct[0]:.1f}%", color=C.PALETTE['old'], fontsize=7, transform=ax.transAxes)

#             # ax.text(.80, .5, f"{percentage_info['percentage'].loc['young']:.1f}", fontsize=6, color=palette['young'], transform=ax.transAxes)
#             # ax.text(.80, .4, f"{percentage_info['percentage'].loc['old']:.1f}", fontsize=6, color=palette['old'], transform=ax.transAxes)
#         ax.axvline(x=C.RT_CUTOFF[1],  lw=0.5, ls='--', alpha=0.8, c='gray')
#         ax.axvline(x=C.RT_CUTOFF[0],  lw=0.5, ls='--', alpha=0.8, c='gray')

#     # sns.despine(trim=True)
#     sns.despine(offset=2, trim=False, ax=ax)
#     if save_fig:
#         fname = f"f1_distribution_{'easy_trials_' if easy_trials else ''}rt_raw_2groups_2025.pdf"
#         save_figure(fig, C.FIGPATH / fname)
#     plt.show()

def plot_rt_distribution(trials_table, clean_rt=False, save_fig=True, easy_trials=False):
    """Histogram of raw RT by age group with cutoffs annotated.

    Non-default settings
    --------------------
    clean_rt : bool
        Keep False here (raw RT view). Cleaning is controlled by filter_trials.
    easy_trials : bool
        If True, include only contrast >= 50%.
    """
    trials = filter_trials(
        trials_table,
        exclude_nan_event_trials=True,
        trial_type=C.TRIAL_TYPE,
        event_list=C.EVENT_LIST,
        clean_rt=False,  # raw distribution view
        rt_variable=C.RT_VARIABLE_NAME,
        rt_cutoff=C.RT_CUTOFF,
    )
    # safety copy to avoid SettingWithCopy warnings
    trials = trials.copy()
    trials = add_age_group(trials)

    data2plot = trials.loc[trials["signed_contrast"].abs() >= 50].copy() if easy_trials else trials.copy()

    # bin labels & categories on a copy
    low, high = C.RT_CUTOFF
    bin_labels = [f"< {int(low*1000)}ms", f"{int(low*1000)}ms - {int(high)}s", f"> {int(high)}s"]
    # safeguard min/max bounds
    lo_bound = float(data2plot["rt_raw"].min())
    hi_bound = float(data2plot["rt_raw"].max())
    # ensure bounds bracket cutoffs
    lo_bound = min(lo_bound, low - 1e-6)
    hi_bound = max(hi_bound, high + 1e-6)

    data2plot.loc[:, "rt_raw_category"] = pd.cut(
        data2plot["rt_raw"], bins=[lo_bound, low, high, hi_bound], labels=bin_labels, right=True
    )
    # cap extreme values for plotting aesthetics only
    data2plot.loc[data2plot["rt_raw"] > high, "rt_raw"] = high + 0.08

    # percentage of the slowest bin within each group
    counts = (data2plot.groupby(["age_group", "rt_raw_category"], as_index=False)["trial_index"]
                        .count()
                        .rename(columns={"trial_index": "n"}))
    totals = (data2plot.groupby("age_group", as_index=False)["trial_index"]
                        .count()
                        .rename(columns={"trial_index": "N_total"}))
    perc = counts.merge(totals, on="age_group", how="left")
    # avoid divide-by-zero
    perc["percentage"] = np.where(perc["N_total"] > 0, (perc["n"] / perc["N_total"]) * 100.0, np.nan)
    slow_bin = bin_labels[-1]
    perc_slow = perc.loc[perc["rt_raw_category"] == slow_bin, ["age_group", "percentage"]].set_index("age_group")

    # choose xlabel based on variable
    if C.RT_VARIABLE_NAME == "response_times_from_stim":
        xlabel = "Response time (s)"
    elif C.RT_VARIABLE_NAME == "firstMovement_times_from_stim":
        xlabel = "Movement initiation time (s)"
    else:
        xlabel = "RT (s)"

    g = sns.FacetGrid(data=data2plot, height=2.36, aspect=1)
    g.map_dataframe(
        sns.histplot,
        x="rt_raw",
        hue="age_group",
        binwidth=0.08,
        stat="probability",
        multiple="dodge",
        common_norm=False,
        palette=C.PALETTE,
        shrink=0.85,
    )
    g.add_legend()

    # annotate cutoffs and slow-bin percentages
    for axidx, ax in enumerate(g.axes.flat):
        ax.set(xlabel=xlabel, xlim=[0, high + 0.05])
        # percentages on the first axis (one panel)
        if axidx == 0:
            yv = perc_slow.reindex(["young", "old"])  # enforce order if both exist
            if "young" in yv.index and pd.notna(yv.loc["young", "percentage"]):
                ax.text(0.80, 0.50, f"{yv.loc['young', 'percentage']:.1f}%",
                        color=C.PALETTE["young"], fontsize=7, transform=ax.transAxes)
            if "old" in yv.index and pd.notna(yv.loc["old", "percentage"]):
                ax.text(0.80, 0.40, f"{yv.loc['old', 'percentage']:.1f}%",
                        color=C.PALETTE["old"], fontsize=7, transform=ax.transAxes)

        ax.axvline(x=high, lw=0.5, ls="--", alpha=0.8, c="gray")
        ax.axvline(x=low,  lw=0.5, ls="--", alpha=0.8, c="gray")
        sns.despine(offset=2, trim=False, ax=ax)

    if save_fig:
        fname = f"f1_distribution_{'easy_trials_' if easy_trials else ''}rt_raw_2groups_2025.pdf"
        save_figure(g, C.FIGPATH / fname)
    plt.show()



# def plot_psychometric_curves(trials_table, save_fig=True):
#     """Plot psychometric curves for each age group."""
#     data2fit = filter_trials(
#         trials_table, exclude_nan_event_trials=True,
#         trial_type=C.TRIAL_TYPE, event_list=C.EVENT_LIST,
#         clean_rt=True, rt_variable=C.RT_VARIABLE_NAME,
#         rt_cutoff=C.RT_CUTOFF
#     )
#     data2fit['age_group'] = (data2fit['mouse_age'] > C.AGE_GROUP_THRESHOLD).map({True: "old", False: "young"})

#     fig = sns.FacetGrid(data2fit, hue="age_group", palette=C.PALETTE, height=2.36, aspect=1)
#     fig.map(plot_psychometric, "signed_contrast", "response", "eid", linewidth=2)
#     fig.despine(trim=True)
#     fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')

#     num_sessions = data2fit.groupby('age_group')['eid'].nunique().to_dict()
#     txt_young = f"Young, " + fr" $n_{{\mathrm{{sessions}}}} = {num_sessions['young']} $"
#     txt_old = f"Old, " +fr" $n_{{\mathrm{{sessions}}}} ={num_sessions['old']} $"

#     for ax in fig.axes.flat:
#         ax.text(0.02, 0.95, txt_young, transform=ax.transAxes, fontsize=7, linespacing=0.8,  c=C.PALETTE['young'], 
#                 verticalalignment='top')
#         ax.text(0.02, 0.88, txt_old, transform=ax.transAxes, fontsize=7, linespacing=0.8, c=C.PALETTE['old'], 
#                 verticalalignment='top')
#     if save_fig:
#         save_figure(fig, C.FIGPATH / "f1_psychfuncs_allmice_first400trials_cleantrials2025.pdf")
def plot_psychometric_curves(trials_table: pd.DataFrame, save_fig: bool = True):
    """Plot psychometric curves for each age group (session-level overlay)."""
    data2fit = filter_trials(
        trials_table,
        exclude_nan_event_trials=True,
        trial_type=C.TRIAL_TYPE,
        event_list=C.EVENT_LIST,
        clean_rt=True,
        rt_variable=C.RT_VARIABLE_NAME,
        rt_cutoff=C.RT_CUTOFF,
    )
    data2fit = add_age_group(data2fit)

    g = sns.FacetGrid(data2fit, hue="age_group", palette=C.PALETTE, height=2.36, aspect=1)
    g.map(plot_psychometric, "signed_contrast", "response", "eid", linewidth=2)
    g.despine(trim=True)
    g.set_axis_labels("Signed contrast (%)", "Rightward choice (%)")

    num_sessions = data2fit.groupby("age_group")["eid"].nunique().to_dict()
    txt_y = f"Young, " + r"$n_{\mathrm{sessions}}=" + f"{num_sessions.get('young', 0)}$"
    txt_o = f"Old, "   + r"$n_{\mathrm{sessions}}=" + f"{num_sessions.get('old', 0)}$"

    for ax in g.axes.flat:
        ax.text(0.02, 0.95, txt_y, transform=ax.transAxes, fontsize=7,
                linespacing=0.8, color=C.PALETTE["young"], va="top")
        ax.text(0.02, 0.88, txt_o, transform=ax.transAxes, fontsize=7,
                linespacing=0.8, color=C.PALETTE["old"], va="top")

    if save_fig:
        save_figure(g, C.FIGPATH / "f1_psychfuncs_allmice_first400trials_cleantrials2025.pdf")



def plot_chronometric_curves(trials_table: pd.DataFrame, clean_rt: bool = True,
                             rt_type: str = "rt", save_fig: bool = True):
    """Plot chronometric curves (median RT or other RT-like vars) by age group."""
    data2fit = filter_trials(
        trials_table,
        exclude_nan_event_trials=True,
        trial_type=C.TRIAL_TYPE,
        event_list=C.EVENT_LIST,
        clean_rt=clean_rt,
        rt_variable=C.RT_VARIABLE_NAME,
        rt_cutoff=C.RT_CUTOFF,
    )
    data2fit = add_age_group(data2fit)

    g = sns.FacetGrid(data2fit, hue="age_group", palette=C.PALETTE,
                      hue_order=["young", "old"], height=2.36, aspect=1)
    g.map(plot_chronometric, "signed_contrast", rt_type, "eid", estimator="median",
          palette=C.PALETTE, linewidth=2)

    for ax in g.axes.flat:
        if rt_type == "rt":
            ax.set(ylim=(0, 0.8))
            ax.set_ylabel("Median RT (s)")
        else:
            ax.set(ylim=(0, 1.5))

    if C.RT_VARIABLE_NAME == "response_times_from_stim":
        g.set_axis_labels("Signed contrast (%)", "Median RT (s)")
    elif C.RT_VARIABLE_NAME == "firstMovement_times_from_stim":
        g.set_axis_labels("Signed contrast (%)", "Movement initiation time(s)")
    else:
        g.set_axis_labels("Signed contrast (%)", "Median RT (s)")

    g.despine(trim=True)

    if save_fig:
        save_figure(g, C.FIGPATH / f"f1_chronfuncs_allmice_first400trials_{rt_type}2025.pdf")


# def plot_chronometric_curves(trials_table, clean_rt=True, rt_type='rt', save_fig=True):
#     """Plot chronometric curves for each age group."""
#     data2fit = filter_trials(
#         trials_table, exclude_nan_event_trials=True,
#         trial_type=C.TRIAL_TYPE, event_list=C.EVENT_LIST,
#         clean_rt=clean_rt, rt_variable=C.RT_VARIABLE_NAME,
#         rt_cutoff=C.RT_CUTOFF
#     )
#     data2fit['age_group'] = (data2fit['mouse_age'] > C.AGE_GROUP_THRESHOLD).map({True: "old", False: "young"})

#     fig = sns.FacetGrid(data2fit, hue="age_group", palette=C.PALETTE, hue_order=['young','old'], height=2.36, aspect=1)
#     fig.map(plot_chronometric, "signed_contrast", rt_type, "eid", estimator='median', palette=C.PALETTE, linewidth=2)
#     for axidx, ax in enumerate(fig.axes.flat):
#         if rt_type =='rt':
#             ax.set(ylim=(0,0.8))
#             ax.set_ylabel('Median RT (s)')
#         else: 
#             ax.set(ylim=(0,1.5))
#     if C.RT_VARIABLE_NAME == 'response_times_from_stim':
#         fig.set_axis_labels('Signed contrast (%)', 'Median RT (s)')
#     elif C.RT_VARIABLE_NAME == 'firstMovement_times_from_stim':
#         fig.set_axis_labels('Signed contrast (%)', 'Movement initiation time(s)')

#     fig.despine(trim=True)

#     if save_fig:
#         save_figure(fig, C.FIGPATH / f"f1_chronfuncs_allmice_first400trials_{rt_type}2025.pdf")

# def plot_chronometric_rt_variability(trials_table, trial_type='first400', clean_rt=True, rt_type = 'rt',
#                                      y_var='rt_CV', save_fig=True):
#     """Plot psychometric functions for each age group."""
#     data2fit = filter_trials(
#         trials_table, exclude_nan_event_trials=True,
#         trial_type=C.TRIAL_TYPE, event_list=C.EVENT_LIST,
#         clean_rt=clean_rt, rt_variable=C.RT_VARIABLE_NAME,
#         rt_cutoff=C.RT_CUTOFF
#     )
#     data2fit['age_group'] = (data2fit['mouse_age'] > C.AGE_GROUP_THRESHOLD).map({True: "old", False: "young"})
#     if y_var == 'rt_CV':
#         data2fit[y_var] = data2fit.groupby(['eid', 'signed_contrast'])[rt_type].transform(
#             lambda x: variation(x.dropna(), ddof=1)
#         )

#     elif y_var == 'rt_MAD':
#         data2fit[y_var] = data2fit.groupby(['eid', 'signed_contrast'])[rt_type].transform(
#             lambda x: stats.median_abs_deviation(x, nan_policy='omit')
#         )
#     data2fit = data2fit.drop_duplicates(subset=['eid', 'signed_contrast', y_var])

#     fig = sns.FacetGrid(data2fit, hue="age_group", palette=C.PALETTE, hue_order=['young','old'],height=2.36, aspect=1)
#     fig.map(plot_chronometric, "signed_contrast", y_var, "eid", estimator='mean', palette=C.PALETTE , linewidth=2)
#     for axidx, ax in enumerate(fig.axes.flat):
 
#         if rt_type == 'rt':
#             ax.set(ylim=(0,0.8))
#         else: 
#             ax.set(ylim=(0,0.4))
#         # ax.set_ylabel('RT variability (CV)')
#     fig.set_axis_labels('Signed contrast (%)', f'RT variability (CV)')

#     fig.despine(trim=True)
    
#     if save_fig:
#         save_figure(fig, C.FIGPATH / f"f1supp_chronfuncs_{y_var}_first400trials_{rt_type}2025.pdf")



def plot_chronometric_rt_variability(trials_table, trial_type="first400",
                                     clean_rt=True, rt_type="rt", y_var="rt_CV", save_fig=True):
    """Chronometric curves for RT variability (CV or MAD) by age group."""
    data2fit = filter_trials(
        trials_table,
        exclude_nan_event_trials=True,
        trial_type=C.TRIAL_TYPE,
        event_list=C.EVENT_LIST,
        clean_rt=clean_rt,
        rt_variable=C.RT_VARIABLE_NAME,
        rt_cutoff=C.RT_CUTOFF,
    ).copy()
    data2fit = add_age_group(data2fit)

    if y_var == "rt_CV":
        data2fit[y_var] = data2fit.groupby(["eid", "signed_contrast"])[rt_type].transform(
            lambda x: variation(x.dropna(), ddof=1)
        )
    elif y_var == "rt_MAD":
        data2fit[y_var] = data2fit.groupby(["eid", "signed_contrast"])[rt_type].transform(
            lambda x: stats.median_abs_deviation(x, nan_policy="omit")
        )

    data2fit = data2fit.drop_duplicates(subset=["eid", "signed_contrast", y_var])

    g = sns.FacetGrid(data2fit, hue="age_group", palette=C.PALETTE,
                      hue_order=["young", "old"], height=2.36, aspect=1)
    g.map(plot_chronometric, "signed_contrast", y_var, "eid", estimator="mean",
          palette=C.PALETTE, linewidth=2)

    for ax in g.axes.flat:
        if rt_type == "rt":
            ax.set(ylim=(0, 0.8))
        else:
            ax.set(ylim=(0, 0.4))

    g.set_axis_labels("Signed contrast (%)", "RT variability (CV)")
    g.despine(trim=True)

    if save_fig:
        save_figure(g, C.FIGPATH / f"f1supp_chronfuncs_{y_var}_first400trials_{rt_type}2025.pdf")



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
        
        txt = format_bf_annotation(beta, p_perm, BF10, BF_conclusion, beta_label="age", big_bf=100)

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
        # if BF10 > 100:
        #     txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}}{mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} > 100, $" + f" {BF_conclusion}"
        # else:
        #     txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}}{mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} = {BF10:.3f}, $" + f" {BF_conclusion}"
        
        txt = format_bf_annotation(beta, p_perm, BF10, BF_conclusion, beta_label="age", big_bf=100)

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
        save_figure(fig, C.FIGPATH / f"f1_correlation_{split_type}_bias_behavioral_perf_paras_psychometric_paras_actualage_2025.pdf")


def plot_rt_paras_scatter(fit_psy_paras_age_info, permut_result_df, save_fig=True):
    measures_list = ['rt_median','rt_CV'] #TODO:rt_MAD,'corr_rt_median','corr_rt_CV','corr_rt_MAD'
    y_labels = ['Median RT (s)','RT variability (CV)']
    for m, measure in enumerate(measures_list):  
        print(f"Processing measure: {measure}")
        try:  
            if measure == 'rt_CV':
                fig, ax =plt.subplots(1, 1, sharex=True, sharey=False, figsize=(2.36, 2.36)) #2.36 for large; 

            else:
                fig, ax =plt.subplots(1, 1, sharex=True, sharey=False, figsize=(1.18, 1.18)) #2.36 for large; 

            beta = permut_result_df[permut_result_df['y_var'] == measure]['observed_val'].values[0]    
            p_perm= permut_result_df[permut_result_df['y_var'] == measure]['p_perm'].values[0]  
            BF10= permut_result_df[permut_result_df['y_var'] == measure]['BF10'].values[0]
            BF_conclusion = permut_result_df[permut_result_df['y_var'] == measure]['BF_conclusion'].values[0]
            
            txt = format_bf_annotation(beta, p_perm, BF10, BF_conclusion, beta_label="age", big_bf=100)

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
                save_figure(fig, C.FIGPATH / f"f1_correlation_behavioral_perf_paras_{measure}_actualage_2025.pdf")

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
        # # - b.histplot: age distribution
        # plot_age_distribution(trials_table,  save_fig=True, session_based=False)
        # # - c.psychometric function 
        # plot_psychometric_curves(trials_table,  save_fig=True)

        # # #TODO:
        # # - d.chronometric function (MEDIAN RT) + scatterplot: Median RT with age
        # plot_chronometric_curves(trials_table,  clean_rt=True,rt_type = 'rt',
        #                      save_fig=True)
        # # - e.chronometric function (RT variabiltiy) 
        # plot_chronometric_rt_variability(trials_table,  clean_rt=True,rt_type = 'rt',
        #                              y_var = 'rt_CV', save_fig=True)

        # #TODO: load fitted results
        split_type = 'prevresp'#'block'
        # eid_unique = trials_table.eid.nunique()
        fit_psy_paras_age_info = read_table(C.RESULTSPATH / f"{split_type}_fit_psy_paras_age_info_367sessions_2025.csv")
        

        # label='psychometric'
        # permut_result_df = read_table(C.RESULTSPATH / f'permutation_test_{label}_{C.AGE2USE}_{C.N_PERMUT_BEHAVIOR}perm_2025.csv')


        # label='shift'
        # permut_result_df = read_table(C.RESULTSPATH / f'permutation_test_{label}_{C.AGE2USE}_{C.N_PERMUT_BEHAVIOR}perm_2025.csv')

        # plot_shift_paras_scatter(fit_psy_paras_age_info, permut_result_df, split_type=split_type, save_fig=True)
        label='rt'
        permut_result_df = read_table(C.RESULTSPATH / f'permutation_test_{label}_{C.AGE2USE}_{C.N_PERMUT_BEHAVIOR}perm_2025.csv')
        # plot_shift_paras_scatter(fit_psy_paras_age_info, permut_result_df, split_type=split_type, save_fig=True)

        # - c. scatter plot:  of 3 psychometric function measures
        plot_psycho_paras_scatter(fit_psy_paras_age_info, permut_result_df, save_fig=True)

        # - f.scatterplot: RT variability with age
        plot_rt_paras_scatter(fit_psy_paras_age_info, permut_result_df, save_fig=True)
        # - g.histplot: raw RT distribution in both groups
        plot_rt_distribution(trials_table,  clean_rt=False, save_fig=True, easy_trials=False)

if __name__ == "__main__":
    main()

# %%
