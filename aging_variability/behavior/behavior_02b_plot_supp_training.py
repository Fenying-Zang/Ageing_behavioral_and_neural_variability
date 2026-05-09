"""

input: data/training_history_149subjs_2025_NEW.parquet
output: figures/Fig1S1_training_history_stats.pdf

Figure: Fig1-S1
1) training time course from start / get_trained
2) performance (easy) on day of get_trained / first_recording
3) #days/sessions/trials until first recording / get_trained

"""
#%%

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import seaborn as sns
import figrid as fg
from ibl_style.utils import get_coords, MM_TO_INCH, double_column_fig
from aging_variability.utils.plot_utils import figure_style

from statsmodels.genmod.families import Gaussian
from aging_variability.utils.data_utils import (
    bf_gaussian_via_pearson, 
    interpret_bayes_factor,
    add_age_group)
from aging_variability.utils.plot_utils import format_bf_annotation
from aging_variability.utils.io import read_table, save_figure
from aging_variability.utils.stats_utils import run_permutation_test
import logging
import aging_variability.config as C

log = logging.getLogger(__name__)
# =====================
# Config 
# =====================
TRAINING_FILE = "training_history_149subjs_2025_NEW.parquet"
TRAINING_AGE_FILE = "training_vs_recording_age_mice.csv"
DAYS_PER_MONTH = 30
N_JOBS = 6
SHUFFLING = "labels1_global"
FAMILY_FUNC = Gaussian()
SAVE_FIGURES = True
# =====================
# 1. Load & Prepare data
# =====================

def prepare_training_table(df):
    """
    Add age-at-training-start variables for age-based analyses.

    Notes
    -----
    - age_months and age_years are based on age at training start.
    - age_group is still defined using age at recording, so the young/old
      grouping matches the main recording-based cohort definition.
    """

    df = df.copy()

    age_table = read_table(C.DATAPATH / TRAINING_AGE_FILE)
    age_table["age_start_months"] = age_table["age_at_start_computed"] / DAYS_PER_MONTH
    age_table["age_recording_months"] = age_table["age_at_recording_computed"] / DAYS_PER_MONTH

    age_cols = [
        "mouse_name",
        "age_at_start_computed",
        "age_at_recording_computed",
        "age_start_months",
        "age_recording_months",
    ]

    df = df.merge(age_table[age_cols], on="mouse_name", how="left")

    df["age_group"] = np.where(
        df["age_at_recording_computed"] < C.AGE_GROUP_THRESHOLD,
        "young",
        "old",
    )

    df["mouse_age"] = df["age_at_start_computed"]
    df["age_months"] = df["age_start_months"]
    df["age_years"] = df["age_months"] / 12.0

    return df


def subset_for_criterion(df, criterion):
    
    """Filter rows with valid offsets for a criterion ('first_recording' or 'get_trained'); return a copy."""

    if criterion == "first_recording":
        return df[~df["num_days_from_recording"].isna()].copy()
    elif criterion == "get_trained":
        return df[~df["num_days_from_trained"].isna()].copy()
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    
def aggregate_until_criterion(df):
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


def fmt_age_annotation(beta, p_perm, data_for_bf, y_col):
    """Compose two-line annotation text (β, p_perm, BF10, conclusion) for a panel; uses BF via pearson and plot_utils.format_bf_annotation."""

    if "age_months" not in data_for_bf.columns:
        data_for_bf = data_for_bf.copy()
        data_for_bf = add_age_group(data_for_bf)

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

    yspans = get_coords(
        height, ratios=[1, 1], space=[25], pad=5, span=(0, 0.6)
    )
    xspans1 = get_coords(
        width, ratios=[0.65, 0.3], space=25, pad=5, span=(0.1, 0.9)
    )
    xspans2 = get_coords(
        width, ratios=[0.65, 0.3], space=25, pad=5, span=(0.1, 0.9)
    )

    axs = {
        "time_course_start": fg.place_axes_on_grid(
            fig, xspan=xspans1[0], yspan=yspans[0]
        ),
        "trained_days": fg.place_axes_on_grid(
            fig, xspan=xspans1[1], yspan=yspans[0]
        ),
        "time_course_trained": fg.place_axes_on_grid(
            fig, xspan=xspans2[0], yspan=yspans[1]
        ),
        "trained_easy_perf": fg.place_axes_on_grid(
            fig, xspan=xspans2[1], yspan=yspans[1]
        ),
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

    else:
        ax.set(xlabel=xlabel, xlim=[-40, 0])

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

def plot_training_until_criterion(training_table, *, criterion, ax, stat_results):
    """
    Scatter panel for number of training days until a criterion.
    """

    df = subset_for_criterion(training_table, criterion)
    df_ag = aggregate_until_criterion(df)

    measure = "num_days"
    y_label = "# training days"

    beta, p_adj, p_perm, sig = extract_stats(stat_results, "y_var", measure)
    txt = fmt_age_annotation(beta, p_perm, df_ag, measure)

    ax.text(0.05, 1, txt, transform=ax.transAxes, fontsize=4)
    scatter_with_age_line(ax, df_ag, measure)

    ax.set_xlabel("Age at training start (months)")
    ax.set_ylabel(y_label)

    return ax


def plot_performance_at_criterion(training_table, *, criterion, n_day_from_criterion,
                                  ax, stat_results):
    
    """Scatter of 'perf_easy' at specific day relative to criterion; annotate with β/p_perm/BF; returns ax."""


    df = training_table.copy()
    df = add_age_group(df)

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
    ax.set_xlabel("Age at training start (months)")
    ax.set_ylabel("Performance\n on easy trials")
    return ax


def plot_performance_from_start(training_table, *, n_day_from_start, ax, stat_results):
    
    """Scatter of 'perf_easy' on a fixed day from start (e.g., 20/50); annotate with β/p_perm/BF; returns ax."""

    df = training_table.copy()
    df = add_age_group(df)
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

    # filename = C.RESULTSPATH / f"training_until_each_criterion_{C.N_PERMUT_BEHAVIOR}permutation.csv"
    filename = C.RESULTSPATH / f"revision_training_until_each_criterion_{C.N_PERMUT_BEHAVIOR}permutation.csv"
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

    # filename = C.RESULTSPATH / f"training_perf_at_criterion_{C.N_PERMUT_BEHAVIOR}permutation.csv"
    filename = C.RESULTSPATH / f"revision_training_perf_at_criterion_{C.N_PERMUT_BEHAVIOR}permutation.csv"
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
    plot_training_comparison_group_mean(
        training_table=training,
        x="num_days_from_start",
        alignment="start",
        ax=axs["time_course_start"],
    )

    plot_training_comparison_group_mean(
        training_table=training,
        x="num_days_from_trained",
        alignment="get_trained",
        ax=axs["time_course_trained"],
    )

    # Stats and panel: number of training days until get_trained
    res_until = stats_until_each_criterion(training)
    plot_training_until_criterion(
        training,
        criterion="get_trained",
        ax=axs["trained_days"],
        stat_results=res_until[res_until["criterion"] == "get_trained"],
    )

    # Stats and panel: performance on get_trained day
    res_at0 = stats_perf_at_criterion(training)
    plot_performance_at_criterion(
        training,
        criterion="get_trained",
        n_day_from_criterion=0,
        ax=axs["trained_easy_perf"],
        stat_results=res_at0,
    )

    # Finalize
    if SAVE_FIGURES:
        os.makedirs(C.FIGPATH, exist_ok=True)
        # save_figure(fig,C.FIGPATH / "Fig1S1_training_history_stats_test.pdf",add_timestamp=True)
        save_figure(fig, C.FIGPATH / "revision_Fig1S1_training_history_stats.pdf", add_timestamp=True)

if __name__ == "__main__":
    from aging_variability.utils.io import setup_logging
    setup_logging()
    main()


# %%
