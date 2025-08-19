"""
behavior_06_plot_supp_rt_variations.py

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

from joblib import Parallel, delayed
from tqdm import tqdm
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import Log

from scripts.utils.plot_utils import figure_style
from ibl_style.utils import get_coords, MM_TO_INCH, double_column_fig
import figrid as fg

# repo utils
from scripts.utils.io import read_table
from scripts.utils.behavior_utils import filter_trials
from scripts.utils.plot_utils import map_p_value
from scripts.utils.data_utils import shuffle_labels_perm, bf_gaussian_via_pearson, interpret_bayes_factor
import config as C

# =====================
# Tunables (script-local)
# =====================
# If you want a quick test run, uncomment next line to override config:
# n_permut_behavior = 20

MEASURES = ['mad_rt', 'cv_rt', 'sd_log_rt']
Y_LABELS = {'mad_rt': 'MAD of RT', 'cv_rt': 'CV of RT', 'sd_log_rt': 'SD of log RT'}
FAMILY_FUNC = Gaussian()
N_JOBS = 6
SHUFFLING = 'labels1_global'  # same as your other scripts


# =====================
# 1) Loading
# =====================
# def load_trials_table(path_or_pathlike) -> pd.DataFrame:
#     df = pd.read_csv(path_or_pathlike)
#     print(f"{df['eid'].nunique()} sessions loaded")
#     return df


# =====================
# 2) Prepare data (filter + variability summary)
# =====================
def filter_for_rt(trials, rt_variable_name, exclude_nan_event_trials=True, clean_rt=True):
    """Apply your standard trial filtering for a given RT definition."""
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
    """Group by session and compute variability metrics."""
    keys = ['eid', 'mouse_age', 'age_months', 'age_years', 'age_group']
    out = filtered_df.groupby(keys, as_index=False).apply(compute_subject_measures).reset_index(drop=True)
    print(f"{len(out)} sessions with variability measures")
    return out


# =====================
# 3) Permutation testing
# =====================
def _single_permutation(i, data, permuted_label, formula, family_func):
    try:
        shuffled = data.copy()
        shuffled[C.AGE2USE] = permuted_label
        model = glm(formula=formula, data=shuffled, family=family_func).fit()
        return model.params[C.AGE2USE]
    except Exception:
        return np.nan


def run_permutation_test(data: pd.DataFrame,
                         age_labels: np.ndarray,
                         formula: str,
                         family_func,
                         shuffling: str,
                         n_permut: int,
                         n_jobs: int,
                         random_state: int = 123):
    permuted_labels, _ = shuffle_labels_perm(
        labels1=age_labels, labels2=None, shuffling=shuffling,
        n_permut=n_permut, random_state=random_state, n_cores=min(4, n_jobs)
    )
    null_dist = Parallel(n_jobs=n_jobs)(
        delayed(_single_permutation)(i, data, permuted_labels[i], formula, family_func)
        for i in tqdm(range(n_permut), desc=f"Permuting {formula}")
    )
    null_dist = np.asarray(null_dist)
    valid_null = null_dist[~np.isnan(null_dist)]

    model = glm(formula=formula, data=data, family=family_func).fit()
    observed = model.params[C.AGE2USE]
    observed_p = model.pvalues[C.AGE2USE]
    p_perm = (np.sum(np.abs(valid_null) >= np.abs(observed)) + 1) / (len(valid_null) + 1)
    return observed, observed_p, p_perm, valid_null


def get_or_compute_perm_results(rt_variable_name, variability_summary, measures=MEASURES, n_permut=C.N_PERMUT_BEHAVIOR):
    """
    Cache permutation results per RT definition; one row per measure.
    """
    out_csv = C.RESULTSPATH / f"2RT_defs_variability_{rt_variable_name}_{n_permut}permutation_2025.csv"
    if out_csv.exists():
        return pd.read_csv(out_csv)

    rows = []
    for measure in measures:
        formula = f"{measure} ~ {C.AGE2USE}"
        idx = ~np.isnan(variability_summary[measure])
        df_fit = variability_summary.loc[idx].reset_index(drop=True)
        age_vals = df_fit[C.AGE2USE].to_numpy()
        observed, observed_p, p_perm, valid_null = run_permutation_test(
            data=df_fit,
            age_labels=age_vals,
            formula=formula,
            family_func=FAMILY_FUNC,
            shuffling=SHUFFLING,
            n_permut=n_permut,
            n_jobs=N_JOBS,
        )
        print(f"[{rt_variable_name}] {measure}: beta={observed:.4f}, p_perm={p_perm:.4f}")
        rows.append({
            'rt_variable': rt_variable_name,
            'y_var': measure,
            'n_perm': n_permut,
            'formula': formula,
            'observed_val': observed,
            'observed_val_p': observed_p,
            'p_perm': p_perm,
            'ave_null_dist': float(np.nanmean(valid_null))
        })
    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    return res


# =====================
# 4) Plotting
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


def _fmt_stats_text(beta, p_perm, BF10, BF_conclusion):
    """Safe mathtext: no '$$' from concatenation."""
    mapped = map_p_value(p_perm)
    first = rf"$\beta_{{\mathrm{{age}}}} = {beta:.3f},\ p_{{\mathrm{{perm}}}} {mapped}$"
    second = (r"$BF_{10} > 100$" if BF10 > 100 else rf"$BF_{{10}} = {BF10:.3f}$")
    return first + "\n" + second + f" {BF_conclusion}"


def plot_rt_variation_panel(ax, variability_summary, measure, perm_df, rt_variable_name):
    """Scatter/regression + annotation for one metric."""
    # pick the right axis key handled by caller; here just draw
    beta = perm_df.loc[(perm_df['rt_variable'] == rt_variable_name) &
                       (perm_df['y_var'] == measure), 'observed_val']
    p_perm = perm_df.loc[(perm_df['rt_variable'] == rt_variable_name) &
                         (perm_df['y_var'] == measure), 'p_perm']

    beta = float(beta.iloc[0]) if len(beta) else np.nan
    p_perm = float(p_perm.iloc[0]) if len(p_perm) else np.nan

    # Bayes factor using Pearson helper
    BF = bf_gaussian_via_pearson(variability_summary, measure, 'age_months')
    BF10 = BF['BF10']
    BF_concl = interpret_bayes_factor(BF10)

    txt = _fmt_stats_text(beta=beta, p_perm=p_perm, BF10=BF10, BF_conclusion=BF_concl)
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
# 5) Main
# =====================
def main(save_fig: bool = True):
    trials_table_file = C.DATAPATH / "ibl_included_eids_trials_table2025_full.csv"
    trials = read_table(trials_table_file)

    fig, axs = build_figure_layout()

    # Two RT definitions, plotted into different rows
    for rt_var in ['response_times_from_stim', 'firstMovement_times_from_stim']:
        df_rt = filter_for_rt(trials, rt_var, exclude_nan_event_trials=True, clean_rt=True)
        var_sum = make_variability_summary(df_rt)

        # Permutation results (cached per RT definition)
        perm_df = get_or_compute_perm_results(rt_var, var_sum, measures=MEASURES, n_permut=C.N_PERMUT_BEHAVIOR)

        # Route to correct row of panels
        row_prefix = 'response' if rt_var == 'response_times_from_stim' else 'move'
        for measure in MEASURES:
            ax = axs[f"{row_prefix}_{measure}"]
            plot_rt_variation_panel(ax, var_sum, measure, perm_df, rt_var)

    fig.supxlabel('Age (months)', y=0.40)
    fig.supylabel(None)
    plt.tight_layout()

    if save_fig:
        out = Path(C.FIGPATH) / "F1_supp_rt_variations.pdf"
        fig.savefig(out)
        print(f"[OK] saved figure: {out}")


if __name__ == "__main__":
    main(save_fig=True)

