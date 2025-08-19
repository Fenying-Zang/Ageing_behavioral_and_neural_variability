#%%
"""
behavior_05_plot_supp_choice_bias.py

Goal
-----
Plot age effects on choice-bias parameter shifts under two splits:
  - split_type='block'       → block_* panels
  - split_type='prevresp'    → prevresp_* panels

Metrics (y):
  - bias_shift
  - lapselow_shift
  - lapsehigh_shift

Outputs
-------
C.FIGPATH / "F1_supp_choice_bias_shifts.pdf"
C.DATAPATH / "<split>_shift_<age2use>_<nperm>permutation.csv" (cache)
"""

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

from scripts.utils.plot_utils import map_p_value
from scripts.utils.data_utils import shuffle_labels_perm, bf_gaussian_via_pearson, interpret_bayes_factor
import config as C


# =====================
# Tunables (script-local)
# =====================
MEASURES = ['bias_shift', 'lapselow_shift', 'lapsehigh_shift']
Y_LABELS = {'bias_shift': 'Bias shift', 'lapselow_shift': 'Lapselow shift', 'lapsehigh_shift': 'Lapsehigh shift'}
FAMILY_FUNC = Gaussian()
N_JOBS = 6
SHUFFLING = 'labels1_global'
SAVE_FIG = True

# =====================
# 1) Loading / preparing
# =====================
def load_fit_params(split_type):
    """
    Load per-session psychometric parameter shifts with age info.
    Expected CSV columns at minimum:
      ['eid', 'mouse_age', 'age_group'(opt), 'age_months'(opt), 'age_years'(opt)]
      + MEASURES
    """
    csv_file = C.RESULTSPATH / f"{split_type}_fit_psy_paras_age_info_367sessions_2025.csv"
    df = pd.read_csv(csv_file)
    # Add/repair age columns & age_group if missing
    if 'mouse_age' in df.columns:
        if 'age_months' not in df.columns:
            df['age_months'] = df['mouse_age'] / 30.0
        if 'age_years' not in df.columns:
            df['age_years'] = df['mouse_age'] / 365.0
        if 'age_group' not in df.columns:
            df['age_group'] = (df['mouse_age'] > C.AGE_GROUP_THRESHOLD).map({True: 'old', False: 'young'})
    else:
        # If mouse_age missing, ensure age2use + age_months present for plotting/stats
        assert C.AGE2USE in df.columns, f"'{C.AGE2USE}' not found in {csv}"
        if 'age_months' not in df.columns:
            if C.AGE2USE == 'age_years':
                df['age_months'] = df['age_years'] * 12.0
            else:
                raise ValueError("Provide 'age_months' or 'mouse_age' in the input CSV.")
        if 'age_group' not in df.columns:
            if 'mouse_age' in df.columns:
                df['age_group'] = (df['mouse_age'] > C.AGE_GROUP_THRESHOLD).map({True: 'old', False: 'young'})
            else:
                # fallback: split by median age2use if mouse_age truly absent
                df['age_group'] = (df[C.AGE2USE] > df[C.AGE2USE].mean()).map({True: 'old', False: 'young'})
    # Keep only rows with all target measures present
    have = [m for m in MEASURES if m in df.columns]
    missing = set(MEASURES) - set(have)
    if missing:
        raise KeyError(f"Missing expected columns {missing} in {csv}")
    return df


# =====================
# 2 Permutation testing (cached)
# =====================
def _single_permutation(i, data, permuted_label, formula, family_func=Gamma(link=Log())):
    try:
        shuffled = data.copy()
        shuffled[C.AGE2USE] = permuted_label
        model = glm(formula=formula, data=shuffled, family=family_func).fit()
        return model.params[C.AGE2USE]
    except Exception:
        return np.nan


def run_permutation_test(data, age_labels, formula, family_func, shuffling, n_permut, n_jobs, random_state=C.RANDOM_STATE):
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


def get_or_compute_perm_results(split_type, df_in, measures_list=MEASURES,
                                n_permut=C.N_PERMUT_BEHAVIOR):
    """
    Cache permutation results per split_type; one row per measure.
    """
    out_csv = C.RESULTSPATH / f"{split_type}_shift_{C.AGE2USE}_{n_permut}permutation.csv"
    if out_csv.exists():
        return pd.read_csv(out_csv)

    rows = []
    for measure in measures_list:
        formula = f"{measure} ~ {C.AGE2USE}"
        idx = ~np.isnan(df_in[measure])
        df_fit = df_in.loc[idx].reset_index(drop=True)
        age_vals = df_fit[C.AGE2USE].to_numpy()

        observed, observed_p, p_perm, valid_null = run_permutation_test(
            data=df_fit,
            age_labels=age_vals,
            formula=formula,
            family_func=FAMILY_FUNC,
            shuffling=SHUFFLING,
            n_permut=n_permut,
            n_jobs=N_JOBS,
            random_state=C.RANDOM_STATE
        )
        print(f"[{split_type}] {measure}: β={observed:.4f}, p_perm={p_perm:.4f}")
        rows.append({
            'split_type': split_type,
            'y_var': measure,
            'n_perm': n_permut,
            'formula': formula,
            'observed_val': observed,
            'observed_val_p': observed_p,
            'p_perm': p_perm,
            'ave_null_dist': float(np.nanmean(valid_null)),
        })
    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    return res


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
        # block row
        'block_bias_shift':      fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0]),
        'block_lapselow_shift':  fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[0]),
        'block_lapsehigh_shift': fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[0]),
        # prevresp row
        'prevresp_bias_shift':      fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1]),
        'prevresp_lapselow_shift':  fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[1]),
        'prevresp_lapsehigh_shift': fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[1]),
    }
    return fig, axs


def _fmt_stats_text(beta, p_perm, BF10, BF_conclusion):
    """Safe mathtext (avoid '$$')."""
    mapped = map_p_value(p_perm)
    line1 = rf"$\beta_{{\mathrm{{age}}}} = {beta:.3f},\ p_{{\mathrm{{perm}}}} {mapped}$"
    line2 = (r"$BF_{10} > 100$" if BF10 > 100 else rf"$BF_{{10}} = {BF10:.3f}$")
    return f"{line1}\n{line2} {BF_conclusion}"


def plot_shift_panel(ax, df_plot, measure, perm_df):
    # stats text
    beta = perm_df.loc[perm_df['y_var'] == measure, 'observed_val'].values[0]
    p_perm = perm_df.loc[perm_df['y_var'] == measure, 'p_perm'].values[0]

    BF = bf_gaussian_via_pearson(df_plot, measure, 'age_months')
    BF10 = BF['BF10']
    BF_concl = interpret_bayes_factor(BF10)

    txt = _fmt_stats_text(beta=beta, p_perm=p_perm, BF10=BF10, BF_conclusion=BF_concl)
    ax.text(0.05, 1.0, txt, transform=ax.transAxes, fontsize=4, linespacing=0.8, va='top')

    do_reg = (BF_concl in ('strong H1', 'moderate H1'))
    sns.regplot(data=df_plot, x='age_months', y=measure,
                fit_reg=do_reg, marker='.', color="1",
                line_kws=dict(color="gray"), ax=ax)
    sns.scatterplot(data=df_plot, x='age_months', y=measure,
                    hue='age_group', hue_order=['young', 'old'],
                    marker='.', legend=False, palette=C.PALETTE, ax=ax)
    sns.despine(offset=2, trim=False, ax=ax)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xlabel(None)
    ax.set_ylabel(Y_LABELS.get(measure, measure))


# =====================
# 4) Main
# =====================
def main(save_fig: bool = SAVE_FIG):
    # for testing, use a smaller number of permutations
    # n_permut_behavior = 100  # TODO: FOR TEST
    fig, axs = build_figure_layout()

    for split_type in ['block', 'prevresp']:
        df = load_fit_params(split_type)

        # permutation results (cached per split_type)
        perm_df = get_or_compute_perm_results(
            split_type=split_type,
            df_in=df,
            measures_list=MEASURES,
            n_permut=C.N_PERMUT_BEHAVIOR
        )

        # draw the three measures for this split
        for measure in MEASURES:
            ax = axs[f"{split_type}_{measure}"]
            plot_shift_panel(ax, df, measure, perm_df)
            # small header for the middle column
            if measure == 'lapselow_shift':
                ax.set_title(f"Split by {split_type}", fontsize=8)

    fig.supxlabel('Age (months)', y=0.40)
    plt.tight_layout()

    if save_fig:
        out = Path(C.FIGPATH) / "F1_supp_choice_bias_shifts.pdf"
        fig.savefig(out)
        print(f"[OK] saved figure: {out}")


if __name__ == "__main__":
    main(save_fig=True)