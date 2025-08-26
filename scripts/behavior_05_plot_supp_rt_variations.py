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

