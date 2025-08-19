"""

input: data/training_history_149subjs_2025_NEW.parquet
output: figures/Fig1S1_training_history_stats.pdf

Figure: Fig1-supp 2 — Older mice took longer to learn the task (same protocols)
1) training time course from start / get_trained
2) performance (easy) on day of get_trained / first_recording
3) #days/sessions/trials until first recording / get_trained

"""
#%%
# =====================
# Imports
# =====================
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from joblib import Parallel, delayed
from tqdm import tqdm
import figrid as fg
from ibl_style.utils import get_coords, MM_TO_INCH, double_column_fig
from scripts.utils.plot_utils import figure_style

from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import Log
from scripts.utils.permutation_test import plot_permut_test
from scripts.utils.data_utils import (shuffle_labels_perm, bf_gaussian_via_pearson, interpret_bayes_factor,
                                      add_age_group, add_age_months, add_age_years)
from scripts.utils.plot_utils import format_bf_annotation
from scripts.utils.io import read_table
from scripts.utils.stats_utils import single_permutation, run_permutation_test

import config as C
# =====================
# Config 
# =====================
TRAINING_FILE = "training_history_149subjs_2025_NEW.parquet"
SAVE_FIGURES = True
N_JOBS = 6
SHUFFLING = "labels1_global"
FAMILY_FUNC = Gaussian()

# =====================
# 1. Load & Prepare data
# =====================

def prepare_training_table(df):
    
    """Add 'age_months' (=mouse_age/30), 'age_years' (=mouse_age/365), and 'age_group'; return a copy."""

    df = df.copy()
    df = add_age_months(df)
    df = add_age_years(df)
    df = add_age_group(df)
    return df


def subset_for_criterion(df, criterion):
    
    """Filter rows with valid offsets for a criterion ('first_recording' or 'get_trained'); return a copy."""

    if criterion == "first_recording":
        return df[~df["num_days_from_recording"].isna()].copy()
    elif criterion == "get_trained":
        return df[~df["num_days_from_trained"].isna()].copy()
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    
def aggregate_until_criterion(df) :
    """
    Aggregate per-mouse totals (num_days/sessions/trials) and attach age bins; 
    input df already filtered to the criterion window.
    """
    g = df.groupby("mouse_name")
    out = pd.DataFrame({
        "mouse_name": g.size().index,
        "num_days": g["trials_date"].transform("count").groupby(df["mouse_name"]).first().values,
        "num_sessions": g["n_session"].transform("sum").groupby(df["mouse_name"]).first().values,
        "num_trials": g["n_trials_day"].transform("sum").groupby(df["mouse_name"]).first().values,
    })
    # attach per-mouse age and group (take first)
    meta = df.groupby("mouse_name").agg({
        "age_group": "first", "age_months": "first", "age_years": "first"
    }).reset_index()
    out = out.merge(meta, on="mouse_name", how="left")
    out["age_binned"] = pd.cut(out["age_months"], bins=[0, 3, 8, 11, 14, 18, 23])
    return out

# =====================
# Stats helpers
# =====================
def extract_stats(df, key_col, key_val):
    """Grab observed beta and p-values for a row matching key_col==key_val; 
    returns (beta, p_adj, p_perm, sig) with np.nan when missing."""

    row = df[df[key_col] == key_val]
    if row.empty:
        return np.nan, np.nan, np.nan, np.nan
    beta = row["observed_val"].values[0]
    p_perm = row["p_perm"].values[0]
    p_adj = row["p_corrected"].values[0] if "p_corrected" in row else np.nan
    sig = row["reject"].values[0] if "reject" in row else np.nan
    return beta, p_adj, p_perm, sig


# def single_permutation(i, data, permuted_label, *,
#                        formula, family_func=Gamma(link=Log())) :
#     """Fit GLM once with permuted 'age_years'; return the fitted coefficient for 'age_years' (np.nan on failure)."""

#     try:
#         shuffled = data.copy()
#         shuffled["age_years"] = permuted_label
#         model = glm(formula=formula, data=shuffled, family=family_func).fit()
#         return model.params["age_years"]
#     except Exception as e:
#         print(f"Permutation {i} failed: {e}")
#         return np.nan


# def run_permutation_test(data, age_labels, *, formula,
#                           family_func=FAMILY_FUNC, shuffling=SHUFFLING,
#                           n_permut=C.N_PERMUT_BEHAVIOR, n_jobs=N_JOBS,
#                           random_state=C.RANDOM_STATE, plot=False):
#     """Permutation test for the 'age_years' term in a GLM; returns (observed_beta, glm_p, perm_p, valid_null)."""

#     permuted_labels, _ = shuffle_labels_perm(
#         labels1=age_labels, labels2=None, shuffling=shuffling,
#         n_permut=n_permut, random_state=random_state, n_cores=n_jobs,
#     )
#     null_dist = Parallel(n_jobs=n_jobs)(
#         delayed(single_permutation)(i, data, permuted_labels[i], formula=formula, family_func=family_func)
#         for i in tqdm(range(n_permut))
#     )
#     null_dist = np.asarray(null_dist)
#     valid_null = null_dist[~np.isnan(null_dist)]

#     model_obs = glm(formula=formula, data=data, family=family_func).fit()
#     observed_val = model_obs.params["age_years"]
#     observed_val_p = model_obs.pvalues["age_years"]
#     p_perm = (np.sum(np.abs(valid_null) >= np.abs(observed_val)) + 1) / (len(valid_null) + 1)

#     if plot:
#         plot_permut_test(null_dist=valid_null, observed_val=observed_val, p=p_perm, mark_p=None)

#     return observed_val, observed_val_p, p_perm, valid_null


def fmt_age_annotation(beta, p_perm, data_for_bf, y_col):
    """Compose two-line annotation text (β, p_perm, BF10, conclusion) for a panel; uses BF via pearson and plot_utils.format_bf_annotation."""

    if "age_months" not in data_for_bf.columns:
        data_for_bf = data_for_bf.copy()
        data_for_bf = add_age_months(data_for_bf)

    BF = bf_gaussian_via_pearson(data_for_bf, y_col, "age_months")
    BF10 = BF["BF10"]
    conclusion = interpret_bayes_factor(BF10)
    return format_bf_annotation(beta, p_perm, BF10, conclusion, beta_label="age", big_bf=100)

        
# =====================
# Plotting
# =====================

def build_figure_layout():
    figure_style()
    fig = double_column_fig()
    width, height = fig.get_size_inches() / MM_TO_INCH
    yspans = get_coords(height, ratios=[0.7, 1, 1, 1], space=[15, 25, 25], pad=5, span=(0, 1))
    xspans2 = get_coords(width, ratios=[1, 1], space=25, pad=5, span=(0, 1))
    xspans1 = get_coords(width, ratios=[1, 1, 1, 1], space=[18, 35, 18], pad=20, span=(0, 1))
    xspans3 = get_coords(width, ratios=[1, 1, 1, 1], space=[20, 20, 20], pad=5, span=(0, 1))
    xspans4 = get_coords(width, ratios=[1, 1, 1, 1], space=[20, 20, 20], pad=5, span=(0, 1))

    axs = {
        "time_course_start": fg.place_axes_on_grid(fig, xspan=xspans2[0], yspan=yspans[1]),
        "time_course_trained": fg.place_axes_on_grid(fig, xspan=xspans2[1], yspan=yspans[1]),
        "start_20": fg.place_axes_on_grid(fig, xspan=xspans1[0], yspan=yspans[0]),
        "start_50": fg.place_axes_on_grid(fig, xspan=xspans1[1], yspan=yspans[0]),
        "trained_10": fg.place_axes_on_grid(fig, xspan=xspans1[2], yspan=yspans[0]),
        "trained_5": fg.place_axes_on_grid(fig, xspan=xspans1[3], yspan=yspans[0]),
        "trained_days": fg.place_axes_on_grid(fig, xspan=xspans3[0], yspan=yspans[2]),
        "trained_sessions": fg.place_axes_on_grid(fig, xspan=xspans3[1], yspan=yspans[2]),
        "trained_trials": fg.place_axes_on_grid(fig, xspan=xspans3[2], yspan=yspans[2]),
        "trained_easy_perf": fg.place_axes_on_grid(fig, xspan=xspans3[3], yspan=yspans[2]),
        "f_record_days": fg.place_axes_on_grid(fig, xspan=xspans4[0], yspan=yspans[3]),
        "f_record_sessions": fg.place_axes_on_grid(fig, xspan=xspans4[1], yspan=yspans[3]),
        "f_record_trials": fg.place_axes_on_grid(fig, xspan=xspans4[2], yspan=yspans[3]),
        "f_record_easy_perf": fg.place_axes_on_grid(fig, xspan=xspans4[3], yspan=yspans[3]),
    }
    return fig, axs


def plot_training_comparison_group_mean(training_table, *, x, alignment,
                                        palette=C.PALETTE, ax=None):
    """Group-mean time course of 'perf_easy' by age_group for a given timeline x; draws cutoffs/labels; returns ax."""


    if x == "num_days_from_recording":
        data2plot = training_table[~training_table["num_days_from_recording"].isna()]
    elif x == "num_days_from_start":
        data2plot = training_table.loc[training_table["num_days_from_start"] <= 60]
    elif x == "num_days_from_trained":
        data2plot = training_table[~training_table["num_days_from_trained"].isna()]
    else:
        raise ValueError(f"Unknown x variable: {x}")

    sns.lineplot(
        data=data2plot, x=x, y="perf_easy", hue="age_group", hue_order=["young","old"],
        palette=palette, estimator="mean", errorbar="se", ax=ax, legend=False,
    )
    xlabel = f"Training day from {alignment}"
    ax.set(ylabel="Performance on easy trials", ylim=[0, 1])

    if x == "num_days_from_start":
        num_old = data2plot[data2plot.age_group == "old"]["mouse_name"].nunique()
        num_young = data2plot[data2plot.age_group == "young"]["mouse_name"].nunique()
        ax.set(xlabel=xlabel)
        ax.annotate(f"{num_old} old mice", xy=(1, 0), ha="right", xycoords="axes fraction",
                    xytext=(-10, 16), textcoords="offset points", color=palette['old'], fontsize=7)
        ax.annotate(f"{num_young} young mice", xy=(1, 0), ha="right", xycoords="axes fraction",
                    xytext=(-10, 10), textcoords="offset points", color=palette['young'], fontsize=7)
        ax.axvline(x=20, ls="--", lw=0.5, alpha=0.8, c="gray")
        ax.axvline(x=50, ls="--", lw=0.5, alpha=1, c="gray")
    else:
        ax.set(xlabel=xlabel, xlim=[-40, 0])
        ax.axvline(x=-10, ls="--", lw=0.5, alpha=0.8, c="gray")
        ax.axvline(x=-5, ls="--", lw=0.5, alpha=1, c="gray")
    sns.despine(offset=2, trim=False, ax=ax)
    return ax


def scatter_with_age_line(ax, df, y_col):

    """Scatter of y vs age_months; add regression line only if BF suggests moderate/strong H1; returns ax."""

    # Decide whether to show regression line based on Bayes factor strength
    BF = bf_gaussian_via_pearson(df, y_col, "age_months")
    conclusion = interpret_bayes_factor(BF["BF10"])
    add_line = conclusion in {"strong H1", "moderate H1"}
    sns.regplot(data=df, x="age_months", y=y_col, marker=".", color="1",
                line_kws=dict(color="gray"), fit_reg=add_line, ax=ax)
    sns.scatterplot(data=df, x="age_months", y=y_col, hue="age_group",
                    alpha=1, marker=".", legend=False, palette=C.PALETTE, hue_order=["young","old"], ax=ax)
    sns.despine(offset=2, trim=False, ax=ax)


def plot_training_until_criterion(training_table, *, criterion, axes, stat_results):

    """Three scatter subpanels (#days/#sessions/#trials) until a criterion; annotate with β/p_perm/BF; returns axes."""

    df = subset_for_criterion(training_table, criterion)
    df_ag = aggregate_until_criterion(df)

    measures = ["num_days", "num_sessions", "num_trials"]
    y_lables = ["# days", "# sessions", "# trials"]
    for m, measure in enumerate(measures):
        ax = axes[m]
        beta, p_adj, p_perm, sig = extract_stats(stat_results, "y_var", measure)
        txt = fmt_age_annotation(beta, p_perm, df_ag, measure)
        ax.text(0.05, 1, txt, transform=ax.transAxes, fontsize=4)
        scatter_with_age_line(ax, df_ag, measure)
        if m == 1:
            ax.set_xlabel("Age (months)")
        else:
            ax.set_xlabel(None)
        # ax.set_ylabel(measure)
        ax.set_ylabel(y_lables[m])
    return axes


def plot_performance_at_criterion(training_table, *, criterion, n_day_from_criterion,
                                  ax, stat_results):
    
    """Scatter of 'perf_easy' at specific day relative to criterion; annotate with β/p_perm/BF; returns ax."""


    df = training_table.copy()
    df = add_age_months(df)

    if criterion == "first_recording":
        data_before = df[df["num_days_from_recording"] == n_day_from_criterion]
    elif criterion == "get_trained":
        data_before = df[df["num_days_from_trained"] == n_day_from_criterion]
    else:
        raise ValueError("criterion must be 'first_recording' or 'get_trained'")

    if n_day_from_criterion == 0:
        beta, _, p_perm, _ = extract_stats(stat_results, "criterion", criterion)
    else:
        beta, _, p_perm, _ = extract_stats(stat_results, "num_days_from_recording", n_day_from_criterion)

    txt = fmt_age_annotation(beta, p_perm, data_before, "perf_easy")
    ax.text(0.05, 1, txt, transform=ax.transAxes, fontsize=4)

    scatter_with_age_line(ax, data_before, "perf_easy")
    ax.set_ylim(0.1, 1.1)
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Performance\n on easy trials")
    return ax


def plot_performance_from_start(training_table, *, n_day_from_start, ax, stat_results):
    
    """Scatter of 'perf_easy' on a fixed day from start (e.g., 20/50); annotate with β/p_perm/BF; returns ax."""

    df = training_table.copy()
    df = add_age_months(df)
    data2plot = df[df["num_days_from_start"] == n_day_from_start]
    beta, p_adj, p_perm, sig = extract_stats(stat_results, "n_day_from_start", n_day_from_start)
    txt = fmt_age_annotation(beta, p_perm, data2plot, "perf_easy")
    ax.text(0.05, 1, txt, transform=ax.transAxes, fontsize=4)

    scatter_with_age_line(ax, data2plot, "perf_easy")
    ax.set_xlabel("Age (months)")
    ax.set_ylim(0.16, 1.1)
    ax.set_ylabel("Performance\n on easy trials")
    return ax

# =====================
# Orchestration (stats runs)
# =====================

def stats_until_each_criterion(training_table, *, criteria=("first_recording", "get_trained")) :
    
    """Permutation for totals until each criterion; caches a CSV under C.RESULTSPATH; returns a long stats table."""

    filename = C.RESULTSPATH / f"training_until_each_criterion_{C.N_PERMUT_BEHAVIOR}permutation.csv"
    if filename.exists():
        all_results = read_table(filename)
    else:
        results = {}
        for criter in criteria:
            df = subset_for_criterion(training_table, criter)
            df_ag = aggregate_until_criterion(df)
            result_rows = []
            for m, measure in enumerate(["num_days", "num_sessions", "num_trials"]):
                formula = f"{measure} ~ age_years"
                idxs = ~np.isnan(df_ag[measure])
                df_fit = df_ag[idxs].reset_index(drop=True)
                obs, obs_p, p_perm, valid_null = run_permutation_test(
                    data=df_fit, age_labels=df_fit["age_years"].values, formula=formula,
                    family_func=FAMILY_FUNC, shuffling=SHUFFLING, n_permut=C.N_PERMUT_BEHAVIOR,
                    n_jobs=N_JOBS, random_state=C.RANDOM_STATE + m, plot=False
                )
                result_rows.append({
                    "criterion": criter,
                    "y_var": measure,
                    "n_perm": C.N_PERMUT_BEHAVIOR,
                    "formula": formula,
                    "observed_val": obs,
                    "observed_val_p": obs_p,
                    "p_perm": p_perm,
                    "ave_null_dist": valid_null.mean(),
                    # "null_dist": valid_null,
                })
            res_df = pd.DataFrame(result_rows)
            results[criter] = res_df
        all_results = pd.concat(
            [df.assign(criterion=criter) for criter, df in results.items()],
            ignore_index=True
        )
        all_results.to_csv(filename, index=False)
    return all_results


def stats_perf_at_criterion(training_table, *, criteria=("first_recording", "get_trained")):
    
    """Permutation for 'perf_easy' at day 0 of each criterion; caches CSV; returns a stats table."""

    filename = C.RESULTSPATH / f"training_perf_at_criterion_{C.N_PERMUT_BEHAVIOR}permutation.csv"
    if filename.exists():
        df = read_table(filename)
    else:
        rows = []
        for criter in criteria:
            if criter == "first_recording":
                data0 = training_table[training_table["num_days_from_recording"] == 0]
            else:
                data0 = training_table[training_table["num_days_from_trained"] == 0]
            data0 = data0[~np.isnan(data0["perf_easy"])].reset_index(drop=True)
            obs, obs_p, p_perm, valid_null = run_permutation_test(
                data=data0, age_labels=data0["age_years"].values, formula="perf_easy ~ age_years",
                family_func=FAMILY_FUNC, shuffling=SHUFFLING, n_permut=C.N_PERMUT_BEHAVIOR,
                n_jobs=N_JOBS, random_state=C.RANDOM_STATE, plot=False
            )
            rows.append({
                "criterion": criter,
                "y_var": "perf_easy",
                "n_perm": C.N_PERMUT_BEHAVIOR,
                "formula": "perf_easy ~ age_years",
                "observed_val": obs,
                "observed_val_p": obs_p,
                "p_perm": p_perm,
                "ave_null_dist": valid_null.mean(),
                # "null_dist": valid_null,
            })
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
    return df


def stats_perf_from_start(training_table, *, days_from_start=(20, 50)) :
    
    """Permutation for 'perf_easy' on selected training days; caches CSV; returns a stats table."""

    filename = C.RESULTSPATH / f"training_perf_from_start_{C.N_PERMUT_BEHAVIOR}permutation.csv"
    if filename.exists():
        df = read_table(filename)
    else:
        rows = []
        for day in days_from_start:
            data = training_table[training_table["num_days_from_start"] == day]
            data = data[~np.isnan(data["perf_easy"])].reset_index(drop=True)
            obs, obs_p, p_perm, valid_null = run_permutation_test(
                data=data, age_labels=data["age_years"].values, formula="perf_easy ~ age_years",
                family_func=FAMILY_FUNC, shuffling=SHUFFLING, n_permut=C.N_PERMUT_BEHAVIOR,
                n_jobs=4, random_state=C.RANDOM_STATE, plot=False
            )
            rows.append({
                "n_day_from_start": day,
                "y_var": "perf_easy",
                "n_perm": C.N_PERMUT_BEHAVIOR,
                "formula": "perf_easy ~ age_years",
                "observed_val": obs,
                "observed_val_p": obs_p,
                "p_perm": p_perm,
                "ave_null_dist": valid_null.mean(),
                # "null_dist": valid_null,
            })
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)

    return df


def stats_perf_before_trained(training_table, *, days_from_trained=(-5, -10)) :
    
    """Permutation for 'perf_easy' on days preceding get_trained (e.g., -5, -10); caches CSV; returns a stats table."""

    filename = C.RESULTSPATH / f"training_perf_before_trained_{C.N_PERMUT_BEHAVIOR}permutation.csv"
    if filename.exists():
        df = read_table(filename)
    else:
        rows = []
        for d in days_from_trained:
            data = training_table[training_table["num_days_from_trained"] == d]
            data = data[~np.isnan(data["perf_easy"])].reset_index(drop=True)
            obs, obs_p, p_perm, valid_null = run_permutation_test(
                data=data, age_labels=data["age_years"].values, formula="perf_easy ~ age_years",
                family_func=FAMILY_FUNC, shuffling=SHUFFLING, n_permut=C.N_PERMUT_BEHAVIOR,
                n_jobs=4, random_state=C.RANDOM_STATE + abs(d), plot=False
            )
            rows.append({
                "num_days_from_recording": d,  # kept the original key to maintain downstream compatibility
                "y_var": "perf_easy",
                "n_perm": C.N_PERMUT_BEHAVIOR,
                "formula": "perf_easy ~ age_years",
                "observed_val": obs,
                "observed_val_p": obs_p,
                "p_perm": p_perm,
                "ave_null_dist": valid_null.mean(),
                # "null_dist": valid_null,
            })
        df = pd.DataFrame(rows)

        df.to_csv(filename, index=False)
    return df

# =====================
# Main
# =====================

def main():
    """
    Orchestrate Fig1S1 ('training history') pipeline.

    Input
    -----
    data : C.DATAPATH / "training_history_149subjs_2025_NEW.parquet"

    Output
    ------
    figure : C.FIGPATH / "Fig1S1_training_history_stats.pdf"
    cached stats : multiple CSV files under C.RESULTSPATH (see stats_* docstrings).

    Notes
    -----
    - SAVE_FIGURES controls saving the final figure.
    - Uses global constants: N_JOBS, SHUFFLING, FAMILY_FUNC, C.N_PERMUT_BEHAVIOR, C.RANDOM_STATE.
    """
    training_fp = C.DATAPATH / TRAINING_FILE  # Path
    training_table = read_table(training_fp)
    training = prepare_training_table(training_table)

    # Figure canvas
    fig, axs = build_figure_layout()

    # Time course panels
    plot_training_comparison_group_mean(training_table=training, x="num_days_from_start", alignment="start", ax=axs["time_course_start"])  
    plot_training_comparison_group_mean(training_table=training, x="num_days_from_trained", alignment="get_trained", ax=axs["time_course_trained"])  

    # Stats — until criterion
    res_until = stats_until_each_criterion(training)
    plot_training_until_criterion(training, criterion="get_trained",
                                  axes=[axs["trained_days"], axs["trained_sessions"], axs["trained_trials"]],
                                  stat_results=res_until[res_until["criterion"]=="get_trained"])
    plot_training_until_criterion(training, criterion="first_recording",
                                  axes=[axs["f_record_days"], axs["f_record_sessions"], axs["f_record_trials"]],
                                  stat_results=res_until[res_until["criterion"]=='first_recording'])  

    # Stats — perf at criterion day 0
    res_at0 = stats_perf_at_criterion(training)
    plot_performance_at_criterion(training, criterion="get_trained", n_day_from_criterion=0,
                                  ax=axs["trained_easy_perf"], stat_results=res_at0)
    plot_performance_at_criterion(training, criterion="first_recording", n_day_from_criterion=0,
                                  ax=axs["f_record_easy_perf"], stat_results=res_at0)

    # Stats — perf from start days (20, 50)
    res_from_start = stats_perf_from_start(training, days_from_start=(20, 50))
    plot_performance_from_start(training, n_day_from_start=20, ax=axs["start_20"], stat_results=res_from_start)
    plot_performance_from_start(training, n_day_from_start=50, ax=axs["start_50"], stat_results=res_from_start)

    # Stats — perf before get_trained (-5, -10)
    res_before = stats_perf_before_trained(training, days_from_trained=(-5, -10))
    plot_performance_at_criterion(training, criterion="get_trained", n_day_from_criterion=-5,
                                  ax=axs["trained_5"], stat_results=res_before)
    plot_performance_at_criterion(training, criterion="get_trained", n_day_from_criterion=-10,
                                  ax=axs["trained_10"], stat_results=res_before)

    # Finalize
    plt.show()
    if SAVE_FIGURES:
        os.makedirs(C.FIGPATH, exist_ok=True)
        fig.savefig(os.path.join(C.FIGPATH, "Fig1S1_training_history_stats.pdf"))


if __name__ == "__main__":
    main()




# #%%
# #=========================================================================================
# def setup_fig_axes(fg, MM_TO_INCH, fig=None):
    
#     if fig is None:
#         fig = double_column_fig()
#     figure_style()
#     # Make a double column figure
#     fig = double_column_fig()
#     # Get the dimensions of the figure in mm
#     width, height = fig.get_size_inches() / MM_TO_INCH #180, 170
#     yspans = get_coords(height, ratios=[0.7, 1, 1, 1], space=[15,25,25], pad=5, span=(0, 1))

#     xspans2 = get_coords(width, ratios=[1, 1], space=25, pad=5, span=(0, 1))#from 0-1
#     xspans1 = get_coords(width, ratios=[1, 1, 1, 1], space=[15,35,15], pad=25, span=(0, 1))#from 0-1
#     xspans3 = get_coords(width, ratios=[1, 1, 1, 1], space=[20,20,20], pad=5, span=(0, 1))#from 0-1
#     xspans4 = get_coords(width, ratios=[1, 1, 1, 1], space=[20,20,20], pad=5, span=(0, 1))#from 0-1

#     axs = {'time_course_start': fg.place_axes_on_grid(fig, xspan=xspans2[0], yspan=yspans[1]),
#         'time_course_trained': fg.place_axes_on_grid(fig, xspan=xspans2[1], yspan=yspans[1]),
        
#         'start_20': fg.place_axes_on_grid(fig, xspan=xspans1[0], yspan=yspans[0]),
#         'start_50': fg.place_axes_on_grid(fig, xspan=xspans1[1], yspan=yspans[0]),
#         'trained_10': fg.place_axes_on_grid(fig, xspan=xspans1[2], yspan=yspans[0]),
#         'trained_5': fg.place_axes_on_grid(fig, xspan=xspans1[3], yspan=yspans[0]),
        
#         'trained_days': fg.place_axes_on_grid(fig, xspan=xspans3[0], yspan=yspans[2]),
#         'trained_sessions': fg.place_axes_on_grid(fig, xspan=xspans3[1], yspan=yspans[2]),
#         'trained_trials': fg.place_axes_on_grid(fig, xspan=xspans3[2], yspan=yspans[2]),
#         'trained_easy_perf': fg.place_axes_on_grid(fig, xspan=xspans3[3], yspan=yspans[2]),

#         'f_record_days': fg.place_axes_on_grid(fig, xspan=xspans4[0], yspan=yspans[3]),
#         'f_record_sessions': fg.place_axes_on_grid(fig, xspan=xspans4[1], yspan=yspans[3]),
#         'f_record_trials': fg.place_axes_on_grid(fig, xspan=xspans4[2], yspan=yspans[3]),
#         'f_record_easy_perf': fg.place_axes_on_grid(fig, xspan=xspans4[3], yspan=yspans[3]),
#     }
#     return fig, axs

# # =====================
# # 1. Load & Prepare Data
# # =====================
# def load_training_table():
#     """
#     Load the training history table from a parquet file.
#     """
#     training_file_name = 'training_history_149subjs_2025_NEW.parquet' #new version
#     try:
#         training_table = read_table(os.path.join(C.DATAPATH, training_file_name), engine='pyarrow')
#     except Exception as err:
#         print(err)
#     training_table['age_group'] = training_table['mouse_age'].apply(lambda x: 'old' if x > C.AGE_GROUP_THRESHOLD else 'young')
#     training_table['age_months'] = training_table['mouse_age']/30
#     training_table['age_years'] = training_table['mouse_age']/365

#     return training_table



# # def load_trial_table(filepath):
# #     trials_table = pd.read_csv(filepath)
# #     print(len(set(trials_table['eid'])), 'sessions loaded')
# #     return trials_table

# def plot_training_comparison_group_mean(training_table =None, x=None, alignment=None, 
#                                         palette=None, ax=None):
#     if x == 'num_days_from_recording' :
#         data2plot = training_table[~training_table['num_days_from_recording'].isna()]

#     elif x == 'num_days_from_start':
#         data2plot = training_table.loc[training_table['num_days_from_start']<=60]

#     elif x == 'num_days_from_trained':
#         data2plot = training_table[~training_table['num_days_from_trained'].isna()]


#     sns.lineplot(data = data2plot
#                         , x=x
#                         , y='perf_easy'
#                         , hue ='age_group'
#                         , hue_order= ['young','old']
#                         , palette=palette
#                         , estimator='mean'
#                         , errorbar='se' #None
#                         , ax=ax
#                         , legend=False
#                      #    , linewidth=3
                     
#                         )
                    
#     xlabel = f'Training day from {alignment}'

#     num_old = data2plot[data2plot.age_group == 'old']['mouse_name'].nunique() 
#     num_young = data2plot[data2plot.age_group == 'young']['mouse_name'].nunique() 

# #     for axidx, ax in enumerate(fig.axes.flat):
#     if x == 'num_days_from_start':
#         ax.set(xlabel=xlabel
#                 ,ylabel ='Performance on easy trials'
#               #   ,title='Training history of mice'
#                 ,ylim=[0,1]
#                 )
#         ax.annotate('%s old mice'%str(num_old), xy=(1, 0), ha='right',xycoords='axes fraction',xytext=(-10, 16),
#                         textcoords='offset points', color=palette[1],fontsize=7)
#         ax.annotate('%s young mice'%str(num_young), xy=(1, 0), ha='right',xycoords='axes fraction',xytext=(-10, 10),
#                         textcoords='offset points', color=palette[0],fontsize=7)
        
#         ax.axvline(x = 20, ls='--', lw=0.5, alpha=0.8, c = 'gray') #add an auxiliary line
#         ax.axvline(x = 50, ls='--', lw=0.5, alpha=1, c = 'gray') #add an auxiliary line


#     else: 
#         ax.set(xlabel=xlabel
#                 ,ylabel ='Performance on easy trials'                
#                 ,xlim=[-40,0]
#                 ,ylim=[0,1]
#                 ) #,title='Training history of mice'
#         # ax.annotate('%s old mice'%str(num_old), xy=(1, 0), ha='right',xycoords='axes fraction',xytext=(-10, 20),
#         #             textcoords='offset points', color=mypalette[1],fontsize=10)
#         # ax.annotate('%s young mice'%str(num_young), xy=(1, 0), ha='right',xycoords='axes fraction',xytext=(-10, 10),
#         #             textcoords='offset points', color=mypalette[0],fontsize=10)
#         ax.axvline(x = -10, ls='--', lw=0.5, alpha=0.8, c = 'gray') #add an auxiliary line
#         ax.axvline(x = -5, ls='--', lw=0.5, alpha=1, c = 'gray') #add an auxiliary line
#     sns.despine(offset=2, trim=False, ax=ax)
  
#     return ax
# # def num_star_new(pvalue):
# #     if pvalue < 0.0001:
# #         stars = ' < 0.0001'
# #     elif pvalue < 0.001:
# #         stars = ' < 0.001'
# #     elif pvalue < 0.01:
# #         stars = ' < 0.01'
# #     else:
# #         stars = pvalue
# #     return stars
# def plot_training_until_criterion(training_table=None,criterion = 'first_recording',
#                                   save=False,figpath=None, axes=None, stat_results = None):
#     if criterion == 'first_recording':
#         data_suppfig = training_table[~training_table['num_days_from_recording'].isna()]
#     elif criterion == 'get_trained':
#         data_suppfig = training_table[~training_table['num_days_from_trained'].isna()]
#     #compute how long until 'get trained'/'first recording'? (days/sessions/trials)
#     data_suppfig=data_suppfig.copy()
#     data_suppfig['num_days'] = data_suppfig.groupby('mouse_name')['trials_date'].transform('count')
#     data_suppfig['num_sessions'] = data_suppfig.groupby('mouse_name')['n_session'].transform('sum')
#     data_suppfig['num_trials'] = data_suppfig.groupby('mouse_name')['n_trials_day'].transform('sum')

#     data_suppfig['age_months'] = data_suppfig['mouse_age']/30
#     data_suppfig['age_binned'] = pd.cut(data_suppfig['age_months'], bins=([0, 3, 8, 11, 14, 18, 23]))
#     data_suppfig_plot = data_suppfig[['mouse_name','age_group','num_days','num_sessions','num_trials','age_binned','age_months']].drop_duplicates()
#     # 3 different y variables:
#     y_variable_list = ['num_days','num_sessions','num_trials'] # num_sessions, num_trials
#     #plot
# #     fig, axs =plt.subplots(1,3,sharex=False, sharey=False, figsize=(20, 6),dpi=300)#50
#     current_stat_results = stat_results[stat_results['criterion']==criterion]
#     for m, measure in enumerate(y_variable_list):    
#         ax = axes[m]

#         beta, p_adj, p_perm, sig = extract_stats(current_stat_results, 'y_var', measure)
#         # stars_p_adj = num_star(p_adj)
#         BF_dict = bf_gaussian_via_pearson(data_suppfig_plot, measure, 'age_months')
#         # print(f"BF10 for {content} vs. {age2use}: {BF_dict['BF10']:.3f}, r={BF_dict['r']:.3f}, n={BF_dict['n']}")
#         BF10 = BF_dict['BF10']
#         BF_conclusion = interpret_bayes_factor(BF10)
#         # stars_p_perm = num_star_new(p_perm)
#         mapped_p_value = map_p_value(p_perm)

#         if BF10 > 100:
#             txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} {mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} > 100, $" + f" {BF_conclusion}"
#         else:
#             txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} {mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} = {BF10:.3f}, $" + f" {BF_conclusion}"

#         ax.text(0.05, 1,txt , transform=ax.transAxes, fontsize=4)

#         if BF_conclusion == 'strong H1' or BF_conclusion == 'moderate H1':
#             sns.regplot(data=data_suppfig_plot, x=data_suppfig_plot['age_months'], y=measure, 
#                     marker='.', color="1", line_kws=dict(color="gray"), ax=ax)
#         else:
#             sns.regplot(data=data_suppfig_plot, x=data_suppfig_plot['age_months'], y=measure, 
#                     fit_reg=False, marker='.', color="1", line_kws=dict(color="gray"), ax=ax) #color = .3
#         sns.scatterplot(x=data_suppfig_plot['age_months'], y=measure, data=data_suppfig_plot, hue='age_group',
#                 alpha=1, marker='.', legend=False, palette=C.PALETTE, hue_order=['young','old'], ax=ax)  #marker='o',s=10, 
#        #  ax.tick_params (labelsize =20) 
#         if m==1:
#             ax.set_xlabel('Age (months)')
#         else:
#             ax.set_xlabel(None)

#         ax.set_ylabel(measure)  
#         sns.despine(offset=2, trim=False, ax=ax)

#     return axes

# def plot_performance_before_criterion(training_table=None, criterion = 'first_recording', n_day_from_criterion = -1, 
#                                       ax=None, stat_results = None):

#     training_table['age_months'] = training_table['mouse_age']/30
#     if criterion == 'first_recording':
#         data_before = training_table[training_table['num_days_from_recording']==n_day_from_criterion]
#     elif criterion == 'get_trained':
#         data_before = training_table[training_table['num_days_from_trained']==n_day_from_criterion]


#     if n_day_from_criterion==0:
#         beta, _, p_perm, sig = extract_stats(stat_results, 'criterion', criterion)
#         p_adj=p_perm
#     else:
#         beta, p_adj, p_perm, sig = extract_stats(stat_results, 'num_days_from_recording', n_day_from_criterion)
#     BF_dict = bf_gaussian_via_pearson(data_before, 'perf_easy', 'age_months')
#     BF10 = BF_dict['BF10']
#     BF_conclusion = interpret_bayes_factor(BF10)
#     mapped_p_value = map_p_value(p_perm)

#     if BF10 > 100:
#         txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} {mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} > 100, $" + f" {BF_conclusion}"
#     else:
#         txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} {mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} = {BF10:.3f}, $" + f" {BF_conclusion}"


#     ax.text(.05, 1, txt, fontsize=4, transform=ax.transAxes)
#     if BF_conclusion == 'strong H1' or BF_conclusion == 'moderate H1':
#         sns.regplot(data=data_before, x='age_months', y='perf_easy', 
#                 marker='.', color="1", line_kws=dict(color="gray"), ax=ax)
#     else:
#         sns.regplot(data=data_before, x='age_months', y='perf_easy', 
#                 fit_reg=False, marker='.', color="1", line_kws=dict(color="gray"), ax=ax) #color = .3
#     sns.scatterplot(x='age_months', y='perf_easy', data=data_before, hue='age_group',
#             alpha=1, marker='.',  legend=False, palette=C.PALETTE, hue_order=['young','old'], ax=ax)  #marker='o',s=10,

#     ax.set_ylim(0.1,1.1)
#     ax.set_xlabel('Age (months)')
#     ax.set_ylabel('Performance \non easy trials')  
#     sns.despine(offset=2, trim=False, ax=ax)
#     return ax

# def plot_performance_from_start(training_table=None,  
#                                   n_day_from_start = 10, ax=None, stat_results = None):

#     training_table['age_months'] = training_table['mouse_age']/30

#     data2plot = training_table[training_table['num_days_from_start']==n_day_from_start]

#     beta, _, p_perm, _ = extract_stats(stat_results, 'n_day_from_start', n_day_from_start)
#     BF_dict = bf_gaussian_via_pearson(data2plot, 'perf_easy', 'age_months')
#     BF10 = BF_dict['BF10']
#     BF_conclusion = interpret_bayes_factor(BF10)
#     mapped_p_value = map_p_value(p_perm)

#     if BF10 > 100:
#         txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} {mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} > 100, $" + f" {BF_conclusion}"
#     else:
#         txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} {mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} = {BF10:.3f}, $" + f" {BF_conclusion}"

#     ax.text(.05, 1, txt, fontsize=4, transform=ax.transAxes)

#     if BF_conclusion == 'strong H1' or BF_conclusion == 'moderate H1':
#         sns.regplot(data=data2plot, x='age_months', y='perf_easy', 
#                 marker='.', color="1", line_kws=dict(color="gray"), ax=ax)
#     else:
#         sns.regplot(data=data2plot, x='age_months', y='perf_easy', 
#                 fit_reg=False, marker='.', color="1", line_kws=dict(color="gray"), ax=ax) #color = .3
#     sns.scatterplot(x='age_months', y='perf_easy', data=data2plot, hue='age_group',
#             alpha=1, marker='.', legend=False, palette=C.PALETTE, hue_order=['young','old'], ax=ax)  #marker='o',s=10, 

#     ax.set_xlabel('Age (months)')
#     ax.set_ylim(0.16,1.1)
#     ax.set_ylabel('Performance \non easy trials')  
#     sns.despine(offset=2, trim=False, ax=ax)
#     return ax

# def single_permutation(i, data, permuted_label, formula2use, family_func=Gamma(link=Log())):
#     try:
#         shuffled_data = data.copy()
#         shuffled_data['age_years'] = permuted_label

#         model = glm(formula=formula2use, data=shuffled_data, family=family_func).fit()
        
#         return model.params["age_years"]
#     except Exception as e:
#         print(f"Permutation {i} failed: {e}")
#         return np.nan
# def extract_stats(df, current_var, current_value):
#     row = df[df[current_var] == current_value]
#     if row.empty:
#         return None, None, None, None  #  raise ValueError(f"Key '{key}' not found")

#     beta = row['observed_val'].values[0]
#     p_perm = row['p_perm'].values[0]

#     try:
#         p_adj = row['p_corrected'].values[0]
#         sig = row['reject'].values[0]
#     except:
#         p_adj = np.nan
#         sig = np.nan
#     return beta, p_adj, p_perm, sig

# def run_permutation_test(
#     data, age_labels, formula, family_func,
#     shuffling, n_permut, n_jobs=4, random_state=123, plot=False
# ):
#     permuted_labels, _ = shuffle_labels_perm(
#         labels1=age_labels,
#         labels2=None,
#         shuffling=shuffling,
#         n_permut=n_permut,
#         random_state=random_state,
#         n_cores=n_jobs
#     )

#     # null distribution
#     null_dist = Parallel(n_jobs=n_jobs)(
#         delayed(single_permutation)(i, data, permuted_labels[i], formula, family_func=family_func)
#         for i in tqdm(range(n_permut))
#     )
    
#     # filter failed 
#     null_dist = np.array(null_dist)
#     valid_null = null_dist[~np.isnan(null_dist)]

#     # observed val
#     model_obs = glm(formula=formula, data=data, family=family_func).fit()
#     observed_val = model_obs.params["age_years"]
#     observed_val_p = model_obs.pvalues["age_years"]
#     p_perm = (np.sum(np.abs(valid_null) >= np.abs(observed_val)) + 1) / (len(valid_null) + 1)

#     if plot:
#         plot_permut_test(null_dist=valid_null, observed_val=observed_val, p=p_perm, mark_p=None)

#     return observed_val, observed_val_p, p_perm, valid_null


# if __name__ == "__main__":

#     training_table = load_training_table()
#     print(len(set(training_table['mouse_name'])), 'mice loaded')
#     save_figures = True

#     fig, axs = setup_fig_axes(fg, MM_TO_INCH)
#     #% time course:
#     plot_training_comparison_group_mean(training_table=training_table, x='num_days_from_start', alignment='start', 
#                                         palette=C.PALETTE, ax = axs['time_course_start'] )

#     plot_training_comparison_group_mean(training_table=training_table, x='num_days_from_trained', alignment='get_trained', 
#                                         palette=C.PALETTE,  ax = axs['time_course_trained'] )

# %%
