#%%
"""
Figure: F1-supp — Choice-bias shifts vs age.

Inputs
------
- Per-session parameter shifts with age: C.RESULTSPATH / "{split_type}_fit_psy_paras_age_info_367sessions_2025.csv"
  where split_type ∈ {"block","prevresp"} and columns include:
  ['eid','mouse_age'(days), 'age_group'(opt), 'age_months'(opt), 'age_years'(opt)] + MEASURES
- Cached permutation table (auto): C.RESULTSPATH / "{split_type}_shift_{C.AGE2USE}_{n_perm}permutation.csv"

Output
------
- Figure: C.FIGPATH / "F1_supp_choice_bias_shifts.pdf"

Panels
------
- Rows: split_type in ["block","prevresp"]
- Cols: MEASURES = ["bias_shift","lapselow_shift","lapsehigh_shift"]

Notes
-----
- Permutation runner is unified: scripts.utils.stats_utils.run_permutation_test (called via helper).
- Text annotation via scripts.utils.plot_utils.format_bf_annotation.
- Age helpers: add_age_group/age_months/age_years if missing.
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

from ibl_style.utils import get_coords, MM_TO_INCH, double_column_fig
import figrid as fg

from scripts.utils.plot_utils import figure_style, map_p_value, format_bf_annotation
from scripts.utils.data_utils import (shuffle_labels_perm, bf_gaussian_via_pearson, interpret_bayes_factor, add_age_group)
import config as C
# from scripts.utils.stats_utils import run_permutation_test as _run_perm_utils
from scripts.utils.stats_utils import get_permut_results_table

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
    """Load parameter-shift table for a split_type and ensure age columns/group exist.
    Expects CSV under C.RESULTSPATH and the MEASURES columns. Returns a DataFrame ready for plotting/stats.
    """
    csv_file = C.RESULTSPATH / f"{split_type}_fit_psy_paras_age_info_367sessions_2025.csv"
    df = pd.read_csv(csv_file)
    # Add/repair age columns & age_group if missing
    if 'mouse_age' in df.columns:
        if 'age_group' not in df.columns:
           df = add_age_group(df)
    else:
        # If mouse_age missing, ensure age2use + age_months present for plotting/stats
        assert C.AGE2USE in df.columns, f"'{C.AGE2USE}' not found in {csv_file}"
        if 'age_months' not in df.columns:
            if C.AGE2USE == 'age_years':
                df['age_months'] = df['age_years'] * 12.0
            else:
                raise ValueError("Provide 'age_months' or 'mouse_age' in the input CSV.")
        if 'age_group' not in df.columns:
            if 'mouse_age' in df.columns:
                df = add_age_group(df)
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
# 3) Plotting
# =====================
def build_figure_layout():
    """Create the 2×3 grid canvas (rows = split_type, cols = MEASURES) and return (fig, axs dict)."""

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


def plot_shift_panel(ax, df_plot, measure, perm_df):
    """
    Plot one measure vs age_months with BF/perm annotation; scatter + optional regression line.
    Uses format_bf_annotation for the text and hue by age_group.
    """

    # stats text
    beta = perm_df.loc[perm_df['y_var'] == measure, 'observed_val'].values[0]
    p_perm = perm_df.loc[perm_df['y_var'] == measure, 'p_perm'].values[0]

    BF = bf_gaussian_via_pearson(df_plot, measure, 'age_months')
    BF10 = BF['BF10']
    BF_concl = interpret_bayes_factor(BF10)

    txt = format_bf_annotation(beta, p_perm, BF10, BF_concl, beta_label="age", big_bf=100)
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
def main(save_fig=SAVE_FIG):
    """Load data for both split types → compute/read permutation tables → draw 2×3 panels → save PDF if requested."""

    fig, axs = build_figure_layout()

    for split_type in ['block', 'prevresp']:
        df = load_fit_params(split_type)

        # # permutation results (cached per split_type)
        # perm_df = get_or_compute_perm_results(
        #     split_type=split_type,
        #     df_in=df,
        #     measures_list=MEASURES,
        #     n_permut=C.N_PERMUT_BEHAVIOR
        # )
        out_csv = C.RESULTSPATH / f"t_{split_type}_shift_{C.AGE2USE}_{C.N_PERMUT_BEHAVIOR}permutation.csv"
        perm_df = get_permut_results_table(
            df=df,
            age_col=C.AGE2USE,
            measures=MEASURES,
            family_func=FAMILY_FUNC,
            shuffling=SHUFFLING,
            n_permut=C.N_PERMUT_BEHAVIOR,
            n_jobs=N_JOBS,
            random_state=C.RANDOM_STATE,
            filename=out_csv
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

