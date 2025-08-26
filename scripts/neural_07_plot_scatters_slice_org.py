"""
Figure 3b,d,g
Figure 3 S1b. Regional specificity of pre-stimulus firing rates.
Figure 3 S2. Regional specificity of post-stimulus firing rates.
Figure 3 S4b. Regional specificity of age-related differences in contrast modulation of firing rates.

Figure 4b,d,f,i 
Figure 4 S2b. Regional specificity of pre-stimulus mean-subtracted Fano Factors.
Figure 4 S2c. Regional specificity of post-stimulus mean-subtracted Fano Factors.
Figure 4 S3. Regional specificity of mean-subtracted Fano Factor quenching.
Figure 4 S5b Regional specificity in age-related differences in contrast modulation of the Fano Factor.

Outputs:
- Omnibus scatter figs per metric
- Region slice-org figs per metric

"""

# %%
# =====================
# Imports (cleaned)
# =====================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ibl_style.utils import MM_TO_INCH
import figrid as fg

import config as C
from scripts.utils.io import read_table, save_figure, get_suffix
from scripts.utils.plot_utils import figure_style, create_slice_org_axes, format_bf_annotation
from scripts.utils.data_utils import add_age_group

from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gaussian
from scripts.utils.io import read_table, save_figure
import logging

log = logging.getLogger(__name__)

# =====================
# Helpers: GLM & age fields
# =====================
def def_glm_formula_plot(metric, mean_subtraction=False, log_transform=False):
    """
    Return a Patsy formula (no C() to avoid name collision with config alias).
    - pooled (across regions): include 'cluster_region' on RHS
    - single-region panels: we later strip 'cluster_region'
    """
    yvar = f'log_{metric}' if log_transform else metric
    if metric in ['fr_delta_modulation', 'ff_quench_modulation']:
        return f"{yvar} ~ age_years + cluster_region + n_trials"
    elif metric == 'ff_quench':
        if mean_subtraction:
            return f"{yvar} ~ age_years + cluster_region + n_trials"
        else:
            return f"{yvar} ~ age_years + cluster_region + abs_contrast + n_trials"
    else:
        if mean_subtraction:
            return f"{yvar} ~ age_years + cluster_region + n_trials"
        else:
            return f"{yvar} ~ age_years + cluster_region + abs_contrast + n_trials"


def strip_region_factor(formula):
    """Remove 'cluster_region' term safely from a simple '+ ...' RHS formula."""
    out = formula.replace('+ cluster_region', '')
    out = out.replace('cluster_region +', '')
    out = out.replace('cluster_region', '')
    return out


def ensure_age_years(df):
    """Ensure an 'age_years' column exists from common age columns."""
    if 'age_years' in df.columns:
        return df
    out = df.copy()
    if 'mouse_age' in out.columns:
        out['age_years'] = out['mouse_age'] / 365.0
    elif 'mouse_age_months' in out.columns:
        out['age_years'] = out['mouse_age_months'] / 12.0
    elif 'age_at_recording' in out.columns:
        out['age_years'] = out['age_at_recording'] / 365.0
    else:
        raise ValueError("Need one of ['mouse_age','mouse_age_months','age_at_recording'] to compute age_years.")
    return out


def ensure_age_months(df):
    """Ensure a 'mouse_age_months' column exists (used on x-axis)."""
    if 'mouse_age_months' in df.columns:
        return df
    out = df.copy()
    if 'mouse_age' in out.columns:
        out['mouse_age_months'] = out['mouse_age'] / 30.0
    elif 'age_years' in out.columns:
        out['mouse_age_months'] = out['age_years'] * 12.0
    elif 'age_at_recording' in out.columns:
        out['mouse_age_months'] = out['age_at_recording'] / 30.0
    else:
        raise ValueError("Need one of ['mouse_age','age_years','age_at_recording'] to compute mouse_age_months.")
    return out


def add_log_y_if_needed(df, metric):
    """Add log_<metric> if not present; small epsilon to avoid log(0)."""
    yvar = f'log_{metric}'
    if yvar not in df.columns:
        out = df.copy()
        out[yvar] = np.log(out[metric] + 1e-6)
        return out
    return df


def pooled_marginal_line(model, raw_df, xgrid_years, method='median', region_weights=None):
    """
    Predict a pooled line by marginalizing cluster_region with weights.
    Default weights are proportional to sample counts per region.
    """
    agg = np.nanmedian if method == 'median' else np.nanmean
    regions, counts = np.unique(raw_df['cluster_region'].dropna(), return_counts=True)

    if region_weights is None:
        w = counts / counts.sum()
        region_weights = dict(zip(regions, w))
    else:
        # normalize supplied weights on existing regions
        s = sum(region_weights.get(r, 0) for r in regions)
        region_weights = {r: region_weights.get(r, 0) / s for r in regions}

    base = {'age_years': xgrid_years}
    if 'n_trials' in raw_df.columns:
        base['n_trials'] = agg(raw_df['n_trials'])
    if 'abs_contrast' in raw_df.columns:
        base['abs_contrast'] = agg(raw_df['abs_contrast'])

    yhats = []
    for r in regions:
        new = pd.DataFrame(base)
        new['cluster_region'] = r
        yhats.append(region_weights[r] * model.predict(new).values)

    return np.sum(np.stack(yhats, axis=0), axis=0)


def glm_fit_predict_from_raw(raw_df, metric, mean_subtraction=False, log_transform=False,
                             method='median', pooled=True):
    """
    Fit GLM and return (xgrid_years, yhat).  Behavior:
      - pooled=True  → include 'cluster_region' and marginalize across it
      - pooled=False → strip 'cluster_region' (single-region panel)
    """
    fam = Gaussian()
    formula = def_glm_formula_plot(metric, mean_subtraction, log_transform)

    df_fit = ensure_age_years(raw_df)
    if log_transform:
        df_fit = add_log_y_if_needed(df_fit, metric)
    if 'cluster_region' in df_fit.columns:
        df_fit = df_fit.copy()
        df_fit['cluster_region'] = df_fit['cluster_region'].astype('category')

    if pooled:
        formula_use = formula
    else:
        formula_use = strip_region_factor(formula)

    # eval_env=0 avoids collisions with user-level names
    model = glm(formula=formula_use, data=df_fit, family=fam, eval_env=0).fit()

    xgrid = np.linspace(df_fit['age_years'].min(), df_fit['age_years'].max(), 200)
    if pooled:
        yhat = pooled_marginal_line(model, raw_df=df_fit, xgrid_years=xgrid, method=method, region_weights=None)
    else:
        # single region: just predict on median covariates of that region-data (no region term present)
        base = {'age_years': xgrid}
        if 'n_trials' in df_fit.columns:
            base['n_trials'] = np.nanmedian(df_fit['n_trials']) if method == 'median' else np.nanmean(df_fit['n_trials'])
        if 'abs_contrast' in df_fit.columns and 'abs_contrast' in formula_use:
            base['abs_contrast'] = np.nanmedian(df_fit['abs_contrast']) if method == 'median' else np.nanmean(df_fit['abs_contrast'])
        new = pd.DataFrame(base)
        yhat = model.predict(new).values

    return xgrid, yhat


def get_vmin_vmax(metric):
    """Per-metric y-limits (kept identical to your previous settings)."""
    ranges = {
        'log_pre_fr': (0.5, 4.5), 'log_post_fr': (0.5, 4),
        'fr_delta_modulation': (-10, 20),
        'log_pre_ff': (-0.75, 0.75), 'log_post_ff': (-0.75, 0.75),
        'ff_quench': (-0.6, 0.4), 'ff_quench_modulation': (-1.5, 1.2)
    }
    return ranges.get(metric, (-1, 1))


# =====================
# Plotting: pooled omnibus scatter
# =====================
def plot_scatter_pooled(df, permut_df, BF_df, y_col='pre_fr', estimator='mean',
                        granularity='probe_level', ylim=(None, None),
                        save=True, mean_subtraction=False):
    """
    Pooled scatter (across regions) with a marginal GLM fit line when BF supports H1.
    Text annotation uses unified format_bf_annotation.
    """
    figure_style()
    raw_df = df.copy()

    # Deduplicate for modulation metrics when not mean-subtracted
    if not mean_subtraction and y_col in ['fr_delta_modulation', 'ff_quench_modulation']:
        raw_df = raw_df.drop(columns=['signed_contrast', 'abs_contrast'], errors='ignore')
        raw_df = raw_df.drop_duplicates(subset=['uuids', 'session_pid', 'mouse_age_months', 'age_group', y_col])

    # Ensure axes age columns
    raw_df = ensure_age_years(ensure_age_months(raw_df))

    # stats
    slope_age = permut_df['observed_val'].values[0]
    p_perm = permut_df['p_perm'].values[0]
    BF_conclusion = BF_df.loc[BF_df['metric'] == y_col, 'BF10_age_category'].values[0]
    BF10 = BF_df.loc[BF_df['metric'] == y_col, 'BF10_age'].values[0]

    fig, ax = plt.subplots(1, 1, figsize=(2.36, 2.36))

    # scatter (size encodes #neurons per session)
    if y_col in ['pre_fr', 'post_fr', 'pre_ff', 'post_ff']:
        log_y = True
        raw_df[f'log_{y_col}'] = np.log(raw_df[y_col] + 1e-6)
        agg_df = (raw_df
                  .groupby(['session_pid', 'mouse_age_months', 'age_group'])
                  .agg(number_neurons=('uuids', 'nunique'),
                       **{f'log_{y_col}': (f'log_{y_col}', 'mean')})
                  ).reset_index()
        sns.scatterplot(x='mouse_age_months', y=f'log_{y_col}', data=agg_df,
                        hue='age_group', marker='.', legend=False, s=agg_df['number_neurons'],
                        palette=C.PALETTE, ax=ax)
        vmin, vmax = get_vmin_vmax(f'log_{y_col}')
    else:
        log_y = False
        agg_df = (raw_df
                  .groupby(['session_pid', 'mouse_age_months', 'age_group'])
                  .agg(number_neurons=('uuids', 'nunique'),
                       **{y_col: (y_col, estimator)})
                  ).reset_index()
        agg_df = agg_df.dropna(subset=['mouse_age_months', y_col])
        sns.scatterplot(x='mouse_age_months', y=y_col, data=agg_df,
                        hue='age_group', marker='.', legend=False, s=agg_df['number_neurons'],
                        palette=C.PALETTE, ax=ax)
        vmin, vmax = get_vmin_vmax(y_col)

    # fit line only if BF supports H1
    if BF_conclusion in ('strong H1', 'moderate H1'):
        xgrid_years, yhat = glm_fit_predict_from_raw(
            raw_df=raw_df,
            metric=y_col,
            mean_subtraction=mean_subtraction,
            log_transform=log_y,
            method='median',
            pooled=True
        )
        ax.plot(xgrid_years * 12.0, yhat, color='gray')

    # y axis config
    ax.set_ylim(vmin, vmax)
    if log_y:
        if 'fr' in y_col:
            tick_vals = [1, 2, 4, 8, 16, 32, 64]
        elif 'ff' in y_col:
            tick_vals = [0.5, 1, 2, 4]
        tick_pos = np.log(tick_vals)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels([str(v) for v in tick_vals])
        ax.set_ylim(np.log(min(tick_vals)), np.log(max(tick_vals)))
        ax.set_ylabel(y_col)
    else:
        ax.set_ylabel(y_col)

    # unified stats annotation
    txt = format_bf_annotation(slope_age, p_perm, BF10, BF_conclusion,
                               beta_label="age", big_bf=100)
    ax.text(0.05, 1.0, txt, transform=ax.transAxes, fontsize=5, va='top',
            bbox=dict(facecolor='white', alpha=0.5))

    ax.set_xlabel('Age (months)')
    ax.set_title(f'Omnibus test: {y_col}')
    ax.set_xticks([5, 10, 15, 20])
    sns.despine(offset=2, trim=False, ax=ax)
    plt.tight_layout()

    if save:
        fname = C.FIGPATH / f'Omnibus_{y_col}_{get_suffix(mean_subtraction)}_scatter.pdf'
        save_figure(fig, fname, add_timestamp=True)


# =====================
# Plotting: per-region slice-org panels
# =====================
def plot_scatter_by_region(df, permut_df, BF_df, y_col='pre_fr', estimator='mean',
                           granularity='probe_level', save=True, mean_subtraction=False):
    """
    Slice-org figure with one small panel per region.
    Uses unified format_bf_annotation and single-region GLM line.
    """
    fig, axs = create_slice_org_axes(fg, MM_TO_INCH)
    figure_style()

    raw_df = ensure_age_years(ensure_age_months(df.copy()))

    for region in C.ROIS:
        ax = axs[region]
        sub_df = raw_df[raw_df['cluster_region'] == region].copy()
        if sub_df.empty:
            continue

        # deduplicate where necessary
        if not mean_subtraction and y_col in ['fr_delta_modulation', 'ff_quench_modulation']:
            sub_df = sub_df.drop(columns=['signed_contrast', 'abs_contrast'], errors='ignore')
            sub_df = sub_df.drop_duplicates(subset=['uuids', 'session_pid', 'mouse_age_months', 'age_group', y_col])

        # stats for this region
        sub_perm = permut_df[permut_df['cluster_region'] == region]
        sub_bf = BF_df[BF_df['cluster_region'] == region]
        if sub_perm.empty or sub_bf.empty:
            continue
        slope_age = sub_perm['observed_val'].values[0]
        p_perm = sub_perm['p_perm'].values[0]
        BF_conclusion = sub_bf['BF10_age_category'].values[0]
        BF10 = sub_bf['BF10_age'].values[0]

        # scatter (size encodes #neurons per session)
        if y_col in ['pre_fr', 'post_fr', 'pre_ff', 'post_ff']:
            log_y = True
            sub_df[f'log_{y_col}'] = np.log(sub_df[y_col] + 1e-6)
            agg_df = (sub_df
                      .groupby(['session_pid', 'mouse_age_months', 'age_group'])
                      .agg(number_neurons=('uuids', 'nunique'),
                           **{f'log_{y_col}': (f'log_{y_col}', 'mean')})
                      ).reset_index()
            agg_df = agg_df.dropna(subset=['mouse_age_months', f'log_{y_col}'])
            sns.scatterplot(x='mouse_age_months', y=f'log_{y_col}', data=agg_df,
                            hue='age_group', marker='.', legend=False, s=agg_df['number_neurons'],
                            palette=C.PALETTE, ax=ax)
            vmin, vmax = get_vmin_vmax(f'log_{y_col}')
        else:
            log_y = False
            agg_df = (sub_df
                      .groupby(['session_pid', 'mouse_age_months', 'age_group'])
                      .agg(number_neurons=('uuids', 'nunique'),
                           **{y_col: (y_col, estimator)})
                      ).reset_index()
            agg_df = agg_df.dropna(subset=['mouse_age_months', y_col])
            sns.scatterplot(x='mouse_age_months', y=y_col, data=agg_df,
                            hue='age_group', marker='.', legend=False, s=agg_df['number_neurons'],
                            palette=C.PALETTE, ax=ax)
            vmin, vmax = get_vmin_vmax(y_col)

        # fit line only if BF supports H1
        if BF_conclusion in ('strong H1', 'moderate H1'):
            xgrid_years, yhat = glm_fit_predict_from_raw(
                raw_df=sub_df,
                metric=y_col,
                mean_subtraction=mean_subtraction,
                log_transform=log_y,
                method='median',
                pooled=False  # single-region panels
            )
            ax.plot(xgrid_years * 12.0, yhat, color='gray', lw=0.8)

        # axes cosmetics
        ax.set_ylim(vmin, vmax)
        if log_y:
            if 'fr' in y_col:
                tick_vals = [1, 3, 9, 27]
            elif 'ff' in y_col:
                tick_vals = [0.5, 1, 2, 4]
            tick_pos = np.log(tick_vals)
            ax.set_yticks(tick_pos)
            ax.set_yticklabels([str(v) for v in tick_vals])
            ax.set_ylim(np.log(min(tick_vals)), np.log(max(tick_vals)))
            ax.set_ylabel(y_col)
        else:
            ax.set_ylabel(y_col)

        # unified annotation
        txt = format_bf_annotation(slope_age, p_perm, BF10, BF_conclusion,
                                   beta_label="age", big_bf=100)
        ax.text(0.05, 1.25, txt, transform=ax.transAxes, fontsize=4, va='top', linespacing=0.8)

        sns.despine(offset=2, trim=False, ax=ax)
        if region in ['ACB', 'OLF', 'MBm', 'PO']:
            ax.set_xticks([5, 10, 15, 20])
        else:
            ax.set_xticks([])
        ax.set_xlabel("  ")
        ax.set_ylabel("  ")

    fig.suptitle(f'{granularity} level: {y_col}', font="Arial", fontsize=8)
    fig.supxlabel('Age (months)', font="Arial", fontsize=8).set_y(0.35)

    if save:
        fname = C.FIGPATH / f"supp_slice_org_{y_col}_{granularity}_{get_suffix(mean_subtraction)}_{estimator}_age_relationship_{C.ALIGN_EVENT}_2025.pdf"
        save_figure(fig, fname, add_timestamp=True)


# =====================
# Main
# =====================
def main(mean_subtraction=False):
    """
    Load neural metrics → add age_group (via utils) + ensure age fields →
    read cached permutation/BF tables → draw pooled + per-region panels.
    """
    if mean_subtraction:
        metrics_path = C.DATAPATH / "neural_metrics_summary_meansub.parquet"
        selected_metrics = C.METRICS_WITH_MEANSUB
    else:
        metrics_path = C.DATAPATH / "neural_metrics_summary_conditions.parquet"
        selected_metrics = C.METRICS_WITHOUT_MEANSUB

    print("Loading extracted neural metrics summary...")
    neural_metrics = read_table(metrics_path)
    neural_metrics = add_age_group(neural_metrics)
    # ensure x/y age fields exist even if upstream files omit them
    neural_metrics = ensure_age_months(ensure_age_years(neural_metrics))

    for metric, est in selected_metrics:
        df_permut_path_pooled = C.RESULTSPATH / f"Omnibus_{metric}_{C.N_PERMUT_NEURAL_OMNIBUS}permutation_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_{get_suffix(mean_subtraction)}.csv"
        df_permut_path_region = C.RESULTSPATH / f"Regional_{metric}_{C.N_PERMUT_NEURAL_OMNIBUS}permutation_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_{get_suffix(mean_subtraction)}.csv"
        df_BF_path_pooled = C.RESULTSPATH / f"omnibus_{get_suffix(mean_subtraction)}BFs_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}.csv"
        df_BF_path_region = C.RESULTSPATH / f"regional_{get_suffix(mean_subtraction)}BFs_{C.ALIGN_EVENT}_{metric}_{C.TRIAL_TYPE}.csv"

        df_permut_pooled = read_table(df_permut_path_pooled)
        df_permut_region = read_table(df_permut_path_region)
        df_BF_pooled = read_table(df_BF_path_pooled)
        df_BF_region = read_table(df_BF_path_region)

        print(f"Plotting {metric}...")
        plot_scatter_pooled(neural_metrics, df_permut_pooled, df_BF_pooled,
                            y_col=metric, estimator=est, granularity='probe_level',
                            save=True, mean_subtraction=mean_subtraction)
        plot_scatter_by_region(neural_metrics, df_permut_region, df_BF_region,
                               y_col=metric, estimator=est, granularity='probe_level',
                               save=True, mean_subtraction=mean_subtraction)


if __name__ == "__main__":
    from scripts.utils.io import setup_logging
    setup_logging()
    

    main(mean_subtraction=True)

# %%
