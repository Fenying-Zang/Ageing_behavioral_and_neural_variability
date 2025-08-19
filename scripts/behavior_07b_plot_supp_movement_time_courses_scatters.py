"""
behavior_07_plot_supp_movement_time_courses.py
"""
#%%
# =====================
# Imports
# =====================
from pathlib import Path
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scripts.utils.plot_utils import figure_style
from ibl_style.utils import get_coords, MM_TO_INCH, double_column_fig
import figrid as fg

from joblib import Parallel, delayed
from tqdm import tqdm

from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import Log

import config as C
from scripts.utils.permutation_test import plot_permut_test
from scripts.utils.data_utils import shuffle_labels_perm, bf_gaussian_via_pearson, interpret_bayes_factor
from scripts.utils.plot_utils import map_p_value
from scripts.utils.io import read_table

# =====================
# Constants / Tunables (script-local)
# =====================
N_JOBS = 6
SHUFFLING = 'labels1_global'  # or 'labels1_based_on_2'
FAMILY_FUNC = Gaussian()      # keep as Gaussian like your original
# PLOT_NULL = False             # set True to view null dist plots for each metric
# TOL = float(tolerance)        # from config, but ensure it's a float

# Metrics to analyze
movement_metrics = [
    'ave_wheel_velocity', 'ave_speed_paw_l', 'ave_speed_paw_r', 'ave_speed_nose_tip',
    'cv_wheel_velocity', 'cv_speed_paw_l', 'cv_speed_paw_r', 'cv_speed_nose_tip'
]


# =====================
# Permutation helpers
# =====================
def single_permutation(i, data, permuted_label, formula2use, family_func=Gamma(link=Log())):
    """
    Fit a GLM after substituting permuted age labels; return the age coefficient.
    Returns np.nan on failure so we can filter safely.
    """
    try:
        shuffled = data.copy()
        shuffled['age_years'] = permuted_label
        model = glm(formula=formula2use, data=shuffled, family=family_func).fit()
        return model.params["age_years"]
    except Exception as e:
        # Optional: print(f"Permutation {i} failed: {e}")
        return np.nan


def run_permutation(df_at_timepoint, n_permut, timepoint, family_func=FAMILY_FUNC, shuffling=SHUFFLING,
                    n_jobs=N_JOBS, plot=False, outdir=C.RESULTSPATH):
    """
    Run permutation tests for all movement metrics at a single timepoint.
    Saves a CSV of the results and returns the DataFrame.
    """
    df = df_at_timepoint.copy()
    df['age_years'] = df['age_at_recording'] / 365

    results = []
    for metric in movement_metrics:
        formula = f"{metric} ~ age_years"
        clean_df = df.dropna(subset=[metric])
        clean_df = clean_df[np.isfinite(clean_df[metric])]

        if clean_df.empty:
            # Record an empty result row (keeps downstream robust)
            results.append({
                'timepoint': timepoint, 'y_var': metric, 'n_perm': n_permut,
                'formula': formula, 'observed_val': np.nan, 'observed_val_p': np.nan,
                'p_perm': np.nan, 'ave_null_dist': np.nan, 'null_dist': []
            })
            continue

        age_vals = clean_df['age_years'].values

        permuted_labels, _ = shuffle_labels_perm(
            labels1=age_vals, labels2=None, shuffling=shuffling,
            n_permut=n_permut, random_state=123, n_cores=min(4, n_jobs)
        )

        null_dist = Parallel(n_jobs=n_jobs)(
            delayed(single_permutation)(i, clean_df, permuted_labels[i], formula, family_func=family_func)
            for i in tqdm(range(n_permut), desc=f"Permuting {metric} @ {timepoint}s")
        )
        null_dist = np.asarray(null_dist)
        valid_null = null_dist[~np.isnan(null_dist)]

        # Observed model
        model_obs = glm(formula=formula, data=clean_df, family=family_func).fit()
        observed_val = model_obs.params["age_years"]
        observed_val_p = model_obs.pvalues["age_years"]

        # Two-sided permutation p
        p_perm = (np.sum(np.abs(valid_null) >= np.abs(observed_val)) + 1) / (len(valid_null) + 1)

        if plot:
            plot_permut_test(null_dist=valid_null, observed_val=observed_val, p=p_perm, mark_p=None)

        results.append({
            'timepoint': timepoint,
            'y_var': metric,
            'n_perm': n_permut,
            'formula': formula,
            'observed_val': observed_val,
            'observed_val_p': observed_val_p,
            'p_perm': p_perm,
            'ave_null_dist': float(np.nanmean(valid_null)) if len(valid_null) else np.nan,
            'null_dist': valid_null  # keep for debugging; not ideal in CSV, but you were saving it
        })

    result_df = pd.DataFrame(results)

    out_csv = outdir / f"movement_t_{timepoint}_{n_permut}permut_results.csv"
    result_df.to_csv(out_csv, index=False)

    return result_df


# =====================
# Plotting
# =====================
def build_figure_layout():
    figure_style()
    fig = double_column_fig()
    width, height = fig.get_size_inches() / MM_TO_INCH
    xspans = get_coords(width, ratios=[1, 1, 1, 1], space=20, pad=5, span=(0, 1))
    yspans = get_coords(height, ratios=[1, 1], space=10, pad=5, span=(0, 0.4))

    axs = {
        'ave_wheel_velocity': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0]),
        'ave_speed_paw_l':   fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[0]),
        'ave_speed_paw_r':   fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[0]),
        'ave_speed_nose_tip':fg.place_axes_on_grid(fig, xspan=xspans[3], yspan=yspans[0]),
        'cv_wheel_velocity': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1]),
        'cv_speed_paw_l':    fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[1]),
        'cv_speed_paw_r':    fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[1]),
        'cv_speed_nose_tip': fg.place_axes_on_grid(fig, xspan=xspans[3], yspan=yspans[1]),
    }
    return fig, axs


def extract_timepoint(df, timepoint=0.0, tolerance=C.TOLERANCE):
    """
    Select rows close to a timepoint. Works if the dataframe has either/both
    'time_dlc' and 'time_wheel' columns by accepting rows where EITHER matches.
    """
    df = df.copy()
    mask = np.zeros(len(df), dtype=bool)
    if 'time_dlc' in df.columns:
        mask |= (np.abs(df['time_dlc'] - timepoint) < tolerance)
    if 'time_wheel' in df.columns:
        mask |= (np.abs(df['time_wheel'] - timepoint) < tolerance)
    return df[mask].copy()


def plot_movement_timecourse(final_df, fig, axes, estimator='mean', save_figures=True):
    for key, ax in axes.items():
        # Sessions counts at ~0s (using DLC time if available)
        if 'time_dlc' in final_df.columns:
            session_counter_df = final_df[np.abs(final_df['time_dlc']) < C.TOLERANCE]
        else:
            session_counter_df = final_df[np.abs(final_df['time_wheel']) < C.TOLERANCE]

        num_sessions = session_counter_df.groupby('age_group')['eid'].nunique()

        sns.lineplot(
            data=final_df, x=('time_dlc' if 'time_dlc' in final_df.columns else 'time_wheel'),
            y=key, estimator=estimator, hue='age_group', hue_order=['young', 'old'],
            palette=C.PALETTE, errorbar=('ci', 95), legend=False, ax=ax
        )

        # annotate counts (guard against missing key)
        old_n = int(num_sessions.get('old', 0))
        young_n = int(num_sessions.get('young', 0))
        ax.text(0.7, 0.88, f'{old_n} sessions', transform=ax.transAxes, fontsize=6, c=C.PALETTE['old'])
        ax.text(0.7, 1.00, f'{young_n} sessions', transform=ax.transAxes, fontsize=6, c=C.PALETTE['young'])

        ax.axvline(x=0, lw=0.5, alpha=0.8, c='gray')
        ax.axvspan(-0.1, 0, facecolor='gray', alpha=0.1, edgecolor='none', linewidth=0)
        ax.axvspan(0.16, 0.26, facecolor='gray', alpha=0.1, edgecolor='none', linewidth=0)

        ax.set_xlim(-0.2, 0.8)
        sns.despine(offset=2, trim=False, ax=ax)
        if key.startswith('cv_'):
            ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
        else:
            ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_ylabel(key)

    fig.suptitle('Movement Speed Time Course (wheel + left cam)')
    fig.supxlabel('Time from stim Onset (s)', y=0.4)

    plt.tight_layout()
    plt.show()

    if save_figures:
        out = C.FIGPATH / f"supp_wheel_cam_movement_timecourse_{C.ALIGN_EVENT}_2025-4.pdf"
        fig.savefig(out)


def _fmt_stats_text(beta, mapped_p_value, BF10, BF_conclusion):
    """
    Safe mathtext: avoid creating '$$' across concatenations.
    """
    first = rf"$\beta_{{\mathrm{{age}}}} = {beta:.3f},\ p_{{\mathrm{{perm}}}} {mapped_p_value}$"
    if BF10 > 100:
        second = r"$BF_{10} > 100$"
    else:
        second = rf"$BF_{{10}} = {BF10:.3f}$"
    return first + "\n" + second + f" {BF_conclusion}"


def plot_movement_scatters(df_at_timepoint, result_df, timepoint, fig, axes, save_fig=False):
    for key, ax in axes.items():
        # stats lines
        beta = result_df.loc[result_df['y_var'] == key, 'observed_val']
        p_perm = result_df.loc[result_df['y_var'] == key, 'p_perm']

        beta = float(beta.iloc[0]) if len(beta) else np.nan
        p_perm = float(p_perm.iloc[0]) if len(p_perm) else np.nan
        mapped_p_value = map_p_value(p_perm) if np.isfinite(p_perm) else " = n/a"

        # session-level average of the metric
        y_var = f'sess_ave_{key}'
        df = df_at_timepoint.copy()
        df[y_var] = df.groupby('eid')[key].transform('mean')
        scatter_df = df[['eid', y_var, 'age_at_recording', 'age_group']].drop_duplicates().reset_index(drop=True)
        scatter_df['age_months'] = scatter_df['age_at_recording'] / 30

        # Bayes factor (Pearson-based helper from your utils)
        BF = bf_gaussian_via_pearson(scatter_df, y_var, 'age_months')
        BF10 = BF['BF10']
        BF_concl = interpret_bayes_factor(BF10)

        txt = _fmt_stats_text(beta=beta, mapped_p_value=mapped_p_value, BF10=BF10, BF_conclusion=BF_concl)
        ax.text(0.05, 1.0, txt, transform=ax.transAxes, fontsize=4, va='top', linespacing=0.8)

        # Regression only if moderate/strong H1
        do_reg = (BF_concl in ('strong H1', 'moderate H1'))
        sns.regplot(
            data=scatter_df, x=scatter_df['age_at_recording'] / 30, y=y_var,
            fit_reg=do_reg, marker='.', color="1", line_kws=dict(color="gray"), ax=ax
        )
        sns.scatterplot(
            x=scatter_df['age_at_recording'] / 30, y=y_var, data=scatter_df,
            hue='age_group', marker='.', legend=False, palette=C.PALETTE, ax=ax
        )

        sns.despine(offset=2, trim=False, ax=ax)
        if key.startswith('cv_'):
            ax.set_xticks([5, 10, 15, 20])
        else:
            ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_ylabel(key)

    supxlabel = fig.supxlabel('Age (months)')
    supxlabel.set_y(0.45)
    plt.show()

    if save_fig:
        C.FIGPATH.mkdir(parents=True, exist_ok=True)
        out = C.FIGPATH / f"supp_move_wheel_cam_{C.ALIGN_EVENT}_{timepoint}_2025-4.pdf"
        fig.savefig(out)


# =====================
# Main
# =====================
def main(save_fig=True):
    # Load data (QC-filtered preferred, else raw)
    try:
        df = read_table(C.DATAPATH / "ibl_QCfiltered_wheel_cam_results.parquet")
        print(len(set(df['eid'])), 'sessions remaining (QC-filtered)')
    except FileNotFoundError:
        print('Cannot find the filtered movement data, loading the raw data...')
        df = read_table(C.DATAPATH / "ibl_wheel&dlc_movement_timecourse_2025.parquet")

    # Age group
    df['age_group'] = df['age_at_recording'].map(lambda x: 'old' if x > C.AGE_GROUP_THRESHOLD else 'young')

    # Timecourses across full window
    fig, axes = build_figure_layout()
    plot_movement_timecourse(df, fig, axes, save_figures=True)

    # Point estimates at specific timepoints
    for timepoint in [0, 0.26]:
        df_tp = extract_timepoint(df, timepoint, tolerance=1e-2)  
        results_csv = C.RESULTSPATH / f"movement_t_{timepoint}_{C.N_PERMUT_BEHAVIOR}permut_results.csv"
        if results_csv.exists():
            perm_df = read_table(results_csv)
        else:
            print(f'Running permutation for t={timepoint} ...')
            perm_df = run_permutation(df_tp, C.N_PERMUT_BEHAVIOR, timepoint)

        fig_sc, axes_sc = build_figure_layout()
        plot_movement_scatters(df_tp, result_df=perm_df, timepoint=timepoint, fig=fig_sc, axes=axes_sc, save_fig=save_fig)
        print(f"Saved figure to {C.FIGPATH}/supp_move_wheel_cam_{C.ALIGN_EVENT}_{timepoint}_2025-4.pdf")


if __name__ == "__main__":
    main(save_fig=True)

# %%
