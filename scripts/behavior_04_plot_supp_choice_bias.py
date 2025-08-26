#%%
"""
Figure 1 S3. No age-related differences in block- or history-dependent choice bias.

Inputs
------
- Per-session parameter shifts with age: C.RESULTSPATH / "{split_type}_fit_psy_paras_age_info_367sessions_2025.csv"
  where split_type ∈ {"block","prevresp"} and columns include:
  ['eid','mouse_age'(days), 'age_group'(opt), 'age_months'(opt), 'age_years'(opt)] + MEASURES
- Cached permutation table (auto): C.RESULTSPATH / "{split_type}_shift_{C.AGE2USE}_{n_perm}permutation.csv"

Output
------
- Figure: C.FIGPATH / "F1S3_choice_bias_shifts.pdf"

Panels
------
- Rows: split_type in ["block","prevresp"]
- Cols: MEASURES = ["bias_shift","lapselow_shift","lapsehigh_shift"]

"""
# =====================
# Imports
# =====================
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gaussian
from ibl_style.utils import get_coords, MM_TO_INCH, double_column_fig
import figrid as fg
from scripts.utils.plot_utils import figure_style, format_bf_annotation
from scripts.utils.data_utils import bf_gaussian_via_pearson, interpret_bayes_factor, add_age_group
import config as C
from scripts.utils.stats_utils import get_permut_results_table
from scripts.utils.io import save_figure
import logging

log = logging.getLogger(__name__)
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
# 2) Plotting
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
# 3) Main
# =====================
def main(save_fig=SAVE_FIG):
    """Load data for both split types → compute/read permutation tables → draw 2×3 panels → save PDF if requested."""

    fig, axs = build_figure_layout()

    for split_type in ['block', 'prevresp']:
        df = load_fit_params(split_type)

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
        figname = C.FIGPATH / "F1S3_choice_bias_shifts.pdf"
        save_figure(fig, figname, add_timestamp=True)


if __name__ == "__main__":
    from scripts.utils.io import setup_logging
    setup_logging()

    main(save_fig=True)


#%% see old version below


# #%%  [my version] 
# n_permut_behavior = 100 #TODO: FOR TEST # Number of permutations for behavioral data
# MEASURES = ['bias_shift','lapselow_shift','lapsehigh_shift']
# Y_LABELS =  ['Bias shift','Lapselow shift','Lapsehigh shift']
# FAMILY_FUNC = Gaussian()
# N_JOBS = 6
# SHUFFLING = 'labels1_global'  # same as your other scripts
# PLOT_NULL = False  # set True to view null dist plots (requires utils.permutation_test.plot_permut_test)

# # =====================
# # 4) Plotting
# # =====================
# def build_figure_layout():
#     figure_style()
#     fig = double_column_fig()
#     width, height = fig.get_size_inches() / MM_TO_INCH
#     xspans = get_coords(width, ratios=[1, 1, 1], space=20, pad=5, span=(0.05, 0.95))
#     yspans = get_coords(height, ratios=[1, 1], space=25, pad=5, span=(0, 0.55))

#     axs = {

#         'block_bias_shift':    fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0]),
#         'block_lapselow_shift':     fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[0]),
#         'block_lapsehigh_shift': fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[0]),
#         'prevresp_bias_shift':        fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1]),
#         'prevresp_lapselow_shift':         fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[1]),
#         'prevresp_lapsehigh_shift':     fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[1]),
#     }
#     return fig, axs



# def _fmt_stats_text(beta: float, p_perm: float, BF10: float, BF_conclusion: str) -> str:
#     """Safe mathtext: no '$$' from concatenation."""
#     mapped = map_p_value(p_perm)
#     first = rf"$\beta_{{\mathrm{{age}}}} = {beta:.3f},\ p_{{\mathrm{{perm}}}} {mapped}$"
#     second = (r"$BF_{10} > 100$" if BF10 > 100 else rf"$BF_{{10}} = {BF10:.3f}$")
#     return first + "\n" + second + f" {BF_conclusion}"


# def get_or_compute_perm_results(split_type: str,
#                                 fit_psy_paras_age_info: pd.DataFrame,
#                                 measures_list: list[str] = MEASURES,
#                                 n_permut: int = n_permut_behavior) -> pd.DataFrame:
#     """
#     Cache permutation results per RT definition; one row per measure.
#     """
#     out_csv = Path(C.DATAPATH) / f"{split_type}_shift_{age2use}_{n_permut_behavior}permutation.csv"
#     if out_csv.exists():
#         return pd.read_csv(out_csv)

#     rows = []
#     for measure in measures_list:
#         formula = f"{measure} ~ {age2use}"
#         idx = ~np.isnan(fit_psy_paras_age_info[measure])
#         df_fit = fit_psy_paras_age_info.loc[idx].reset_index(drop=True)
#         age_vals = df_fit[age2use].to_numpy()
#         observed, observed_p, p_perm, valid_null = run_permutation_test(
#             data=df_fit,
#             age_labels=age_vals,
#             formula=formula,
#             family_func=FAMILY_FUNC,
#             shuffling=SHUFFLING,
#             n_permut=n_permut,
#             n_jobs=N_JOBS,
#         )
#         print(f"[{split_type}] {measure}: beta={observed:.4f}, p_perm={p_perm:.4f}")
#         rows.append({
#             'split_type': split_type,
#             'y_var': measure,
#             'n_perm': n_permut,
#             'formula': formula,
#             'observed_val': observed,
#             'observed_val_p': observed_p,
#             'p_perm': p_perm,
#             'ave_null_dist': float(np.nanmean(valid_null))
#         })
#     res = pd.DataFrame(rows)
#     res.to_csv(out_csv, index=False)
#     return res


# def run_permutation_test(data: pd.DataFrame,
#                          age_labels: np.ndarray,
#                          formula: str,
#                          family_func,
#                          shuffling: str,
#                          n_permut: int,
#                          n_jobs: int,
#                          random_state: int = 123):
#     permuted_labels, _ = shuffle_labels_perm(
#         labels1=age_labels, labels2=None, shuffling=shuffling,
#         n_permut=n_permut, random_state=random_state, n_cores=min(4, n_jobs)
#     )
#     null_dist = Parallel(n_jobs=n_jobs)(
#         delayed(_single_permutation)(i, data, permuted_labels[i], formula, family_func)
#         for i in tqdm(range(n_permut), desc=f"Permuting {formula}")
#     )
#     null_dist = np.asarray(null_dist)
#     valid_null = null_dist[~np.isnan(null_dist)]

#     model = glm(formula=formula, data=data, family=family_func).fit()
#     observed = model.params[age2use]
#     observed_p = model.pvalues[age2use]
#     p_perm = (np.sum(np.abs(valid_null) >= np.abs(observed)) + 1) / (len(valid_null) + 1)
#     return observed, observed_p, p_perm, valid_null

# def _single_permutation(i, data, permuted_label, formula, family_func=Gamma(link=Log())):
#     try:
#         shuffled = data.copy()
#         shuffled[age2use] = permuted_label
#         model = glm(formula=formula, data=shuffled, family=family_func).fit()
#         return model.params[age2use]
#     except Exception:
#         return np.nan


# if __name__ == "__main__":
#     save_fig = True

#     fig, axs = build_figure_layout()
#     for split_type in ['prevresp', 'block']:
#         fit_psy_paras_age_info = pd.read_csv(os.path.join(C.DATAPATH, f'{split_type}_fit_psy_paras_age_info_367sessions_2025.csv'))

#         for m, measure in enumerate(MEASURES):  
#             ax = axs[f'{split_type}_{measure}']  

#             #load permut results
#             permut_result_df = get_or_compute_perm_results(
#                 split_type=split_type,
#                 fit_psy_paras_age_info=fit_psy_paras_age_info,
#                 measures_list=MEASURES,
#                 n_permut=n_permut_behavior
#             )

#             beta = permut_result_df[permut_result_df['y_var'] == measure]['observed_val'].values[0]    
#             p_perm= permut_result_df[permut_result_df['y_var'] == measure]['p_perm'].values[0]  
            
#             BF_dict = bf_gaussian_via_pearson(fit_psy_paras_age_info, measure, 'age_months')
#             BF10 = BF_dict['BF10']
#             BF_conclusion = interpret_bayes_factor(BF10)
#             mapped_p_value = map_p_value(p_perm)

#             if BF10 > 100:
#                 txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} {mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} > 100, $" + f" {BF_conclusion}"
#             else:
#                 txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} {mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} = {BF10:.3f}, $" + f" {BF_conclusion}"

#             ax.text(0.05, 1, txt , transform=ax.transAxes, fontsize=4, linespacing=0.8,  
#                     verticalalignment='top')

#             if BF_conclusion == 'strong H1':
#                 sns.regplot(data=fit_psy_paras_age_info, x='age_months', y=measure, 
#                         marker='.', color="1", line_kws=dict(color="gray"), ax=ax)
#             else:
#                 sns.regplot(data=fit_psy_paras_age_info, x='age_months', y=measure, 
#                         fit_reg=False, marker='.', color="1", line_kws=dict(color="gray"), ax=ax) #color = .3
#             sns.scatterplot(x='age_months', y=measure, data=fit_psy_paras_age_info, hue='age_group',
#                     marker='.',legend=False, palette=palette, hue_order=['young','old'], ax=ax)  #marker='o',
            
#             sns.despine(offset=2, trim=False, ax=ax)

#             ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
#             ax.set_xlabel(None)
#             ax.set_ylabel(Y_LABELS[m])  
#             if m==1:
#                 ax.set_title(f"Split by {split_type}", fontsize=8)

#     fig.supxlabel('Age (months)', font="Arial",fontsize=7, y=0.4)

#     plt.tight_layout()
#     if save_fig:
#         fig.savefig(os.path.join(C.FIGPATH, f"T_f1_bias_behavioral_perf_paras_actualage_2025.pdf")) 
