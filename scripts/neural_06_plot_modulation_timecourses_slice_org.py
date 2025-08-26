"""
Figure 3f
Figure 3 S3. Regional specificity in contrast modulation of firing rate.
Figure 3 S4a. Regional specificity of age-related differences in contrast modulation of firing rates
Figure 4h
Figure 4 S4. Regional specificity in contrast modulation of the Fano Factor
Figure 4 S5a Regional specificity in age-related differences in contrast modulation of the Fano Factor
"""
#%%
import config as C
# import warnings
# warnings.simplefilter("default", DeprecationWarning)

import pandas as pd
import os
import matplotlib.pyplot as plt
import figrid as fg
import seaborn as sns

from ibl_style.utils import MM_TO_INCH
from scripts.utils.data_utils import add_age_group
from scripts.utils.plot_utils import figure_style
from scripts.utils.plot_utils import create_slice_org_axes, add_window_label
from scripts.utils.io import read_table
from scripts.utils.io import read_table, save_figure
import logging

log = logging.getLogger(__name__)


def baseline_correct(df, y_col, estimator):
    """Subtract the value at time=0 from all timepoints for each unit/contrast group."""
    if estimator =='mean':
        baseline = df[df['timepoints'].abs() < C.tolerance].groupby(['uuids', 'abs_contrast'])[y_col].mean().rename('baseline')
    else:
        baseline = df[df['timepoints'].abs() < C.tolerance].groupby(['uuids', 'abs_contrast'])[y_col].median().rename('baseline')

    df = df.join(baseline, on=['uuids', 'abs_contrast'])
    df[y_col] = df[y_col] - df['baseline']
    return df.drop(columns='baseline')


def plot_modulation_time_courses_pooled(df, y_col='frs', estimator='median', granularity='neuron_level', split_by_age=False, save=True):
    """
    Plot pooled baseline-corrected timecourses of neural metrics across absolute contrast levels.
    Collapses data across neurons for each contrast level.
    """
    if granularity == 'neuron_level':
        df = df.copy()
        df[y_col] = df.groupby(['uuids', 'timepoints', 'abs_contrast'])[y_col].transform(estimator)
        df = df.drop_duplicates(subset=['uuids', 'timepoints', 'abs_contrast', y_col, 'age_group'])

    df = baseline_correct(df, y_col, estimator)

    figure_style()
    fig, ax = plt.subplots(figsize=(3, 2))

    if split_by_age:

        for group_name, group_df in df.groupby('age_group'):
            if group_df.empty:
                continue
            cmap = C.PALETTE5_2GROUPS.get(group_name, C.PALETTE5)
            sns.lineplot(
                data=group_df, x='timepoints', y=y_col, hue='abs_contrast', ax=ax,
                palette=cmap, lw=0.5, legend=False,
                estimator=estimator, errorbar=None #('ci', 95)
            )
    else:

        sns.lineplot(
            data=df, x='timepoints', y=y_col, lw=0.5,
            estimator=estimator, hue='abs_contrast',
            palette=C.PALETTE5, legend=False, ax=ax, errorbar=('ci', 95)
        )

    ax.axvline(x=0, lw=0.5, alpha=0.8, c='gray')
    ax.axvspan(-0.1, 0, facecolor='gray', alpha=0.1)
    ax.axvspan(0.16, 0.26, facecolor='gray', alpha=0.1)
    add_window_label(ax, -0.1, 0.0,  'pre',  location='outside')
    add_window_label(ax, 0.16, 0.26, 'post', location='outside')
    ax.set_xlim(-0.2, 0.8)
    ax.set_xlabel("Time from stimulus onset (s)", font="Arial", fontsize=8)
    ax.set_ylabel(f'{estimator} {y_col} (baseline corrected)', font="Arial", fontsize=8)

    sns.despine(offset=2, trim=False, ax=ax)
    plt.tight_layout()

    if save:
        suffix = '2groups_' if split_by_age else ''
        fname = C.FIGPATH / f"Omnibus_group_modulation_{granularity}_{suffix}{y_col}_{estimator}_timecourse_{C.ALIGN_EVENT}_noerrorbar.pdf"
        save_figure(fig, fname, add_timestamp=True)


def plot_modulation_time_courses_pooled_2groups(df, y_col='frs', estimator='mean', granularity='neuron_level', save=True):
    """
    Plot pooled baseline-corrected timecourses of neural metrics across absolute contrast levels.
    Collapses data across neurons for each contrast level.
    """
    if granularity == 'neuron_level':
        df = df.copy()
        df[y_col] = df.groupby(['uuids', 'timepoints', 'abs_contrast'])[y_col].transform(estimator)
        df = df.drop_duplicates(subset=['uuids', 'timepoints', 'abs_contrast', y_col, 'age_group'])

    df = baseline_correct(df, y_col, estimator)

    figure_style()
    fig, ax = plt.subplots(figsize=(3, 2))

    sns.lineplot(
        data=df, x='timepoints', y=y_col,lw=0.5,
        estimator=estimator, hue='abs_contrast',
        palette=C.PALETTE5, legend=False, ax=ax, errorbar=('ci', 95),
    )

    ax.axvline(x=0, lw=0.5, alpha=0.8, c='gray')
    ax.axvspan(-0.1, 0, facecolor='gray', alpha=0.1)
    ax.axvspan(0.16, 0.26, facecolor='gray', alpha=0.1)
    
    ax.set_xlim(-0.2, 0.8)
    ax.set_xlabel("Time from stimulus onset (s)", font="Arial", fontsize=8)
    ax.set_ylabel(f'{estimator} {y_col} (baseline corrected)', font="Arial", fontsize=8)
    ax.set_title(f"{granularity}, {y_col}", font="Arial", fontsize=8)

    sns.despine(offset=2, trim=False, ax=ax)
    plt.tight_layout()

    if save:
        fname = C.FIGPATH / f"Omnibus_group_modulation_{granularity}_{y_col}_{estimator}_timecourse_{C.ALIGN_EVENT}.pdf"
        save_figure(fig, fname, add_timestamp=True)


def plot_modulation_time_courses_by_region(df, y_col='frs', estimator='median', granularity='neuron_level', split_by_age=False, save=True):
    """
    Plot regional baseline-corrected timecourses of neural metrics across absolute contrast levels.
    If `split_by_age` is True, plots each age group with separate contrast color maps.
    """
    fig, axs = create_slice_org_axes(fg, MM_TO_INCH)
    figure_style()

    for region in C.ROIS:
        ax = axs[region]
        sub_df = df[df['cluster_region'] == region]
        if sub_df.empty:
            continue

        if granularity == 'neuron_level':
            sub_df = sub_df.groupby(['uuids', 'age_group', 'timepoints', 'abs_contrast'])[y_col].agg(estimator).reset_index()

        
        if split_by_age:
            #baseline correction:
            sub_df = baseline_correct(sub_df, y_col, estimator)

            for group_name, group_df in sub_df.groupby('age_group'):
                if group_df.empty:
                    continue
                cmap = C.PALETTE5_2GROUPS.get(group_name, C.PALETTE5)
                sns.lineplot(
                    data=group_df, x='timepoints', y=y_col, hue='abs_contrast', ax=ax,
                    palette=cmap, lw=0.5, legend=False,
                    estimator=estimator, errorbar=None
                )
        else:
            sns.lineplot(
                data=sub_df, x='timepoints', y=y_col, estimator=estimator, lw=0.5,
                hue='abs_contrast', ax=ax, palette=C.PALETTE5, legend=False, errorbar=None
            )

        ax.axvline(x=0, ls='--', lw=0.5, alpha=0.8, c='gray')
        ax.axvspan(-0.1, 0, facecolor='gray', alpha=0.1)
        ax.axvspan(0.16, 0.26, facecolor='gray', alpha=0.1)
        if region == 'MOs':
            add_window_label(ax, -0.1, 0.0,  'pre',  location='outside', lw=0.7, fontsize=6)
            add_window_label(ax, 0.16, 0.26, 'post', location='outside', lw=0.7, fontsize=6)
        else:
            add_window_label(ax, -0.1, 0.0,  '', location='outside', lw=0.7, fontsize=6)
            add_window_label(ax, 0.16, 0.26, '', location='outside', lw=0.7, fontsize=6)
 
        ax.set_xlim(-0.2, 0.8)

        sns.despine(offset=2, trim=False, ax=ax)
        if region in ['ACB', 'OLF', 'MBm', 'PO']:
            ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
        else:
            ax.set_xticks([])

        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(f'{granularity} {estimator} {y_col} (baseline corrected) time course', fontsize=8)
    if C.ALIGN_EVENT == 'stim':
        fig.supxlabel('Time from stimulus onset (s)', fontsize=8).set_y(0.35)
    else:
        fig.supxlabel('Time from movement onset (s)', fontsize=8).set_y(0.35)

    if save:
        suffix = '2groups_' if split_by_age else ''
        fname = C.FIGPATH / f"Slice_org_modulation_{suffix}{y_col}_timecourse_{C.ALIGN_EVENT}.pdf"
        save_figure(fig, fname, add_timestamp=True)


def main():

    print(f"Loading df_all_conditions ...")
    df_cond_path = C.DATAPATH / f"ibl_BWMLL_FFs_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_conditions_2025.parquet"
    df_cond = read_table(df_cond_path)
    df_cond = add_age_group(df_cond)
    print(f"Plotting omnibus frs contrast modulation time courses...")
    plot_modulation_time_courses_pooled(df_cond, y_col='frs', estimator='mean', granularity='neuron_level',split_by_age=False, save=True)

    print(f"Plotting omnibus FFs contrast modulation time courses...")
    plot_modulation_time_courses_pooled(df_cond, y_col='FFs', estimator='mean', granularity='neuron_level',split_by_age=False, save=True)

    print(f"Plotting frs contrast modulation time courses by regions...")
    plot_modulation_time_courses_by_region(df_cond, y_col='frs', estimator='mean', granularity='neuron_level', split_by_age=False, save=True)

    print(f"Plotting FFs contrast modulation time courses by regions...")
    plot_modulation_time_courses_by_region(df_cond, y_col='FFs', estimator='mean', granularity='neuron_level', split_by_age=False, save=True)

    print(f"Plotting frs contrast modulation time courses by regions...[age group split]")
    plot_modulation_time_courses_by_region(df_cond, y_col='frs', estimator='mean', granularity='neuron_level', split_by_age=True, save=True)

    print(f"Plotting FFs contrast modulation time courses by regions...[age group split]")
    plot_modulation_time_courses_by_region(df_cond, y_col='FFs', estimator='mean', granularity='neuron_level', split_by_age=True, save=True)


    # print(f"Plotting omnibus frs contrast modulation time courses...") # split by age
    # plot_modulation_time_courses_pooled(df_cond, y_col='frs', estimator='mean', granularity='neuron_level',split_by_age=True,  save=True)

    # print(f"Plotting omnibus FFs contrast modulation time courses...") # split by age
    # plot_modulation_time_courses_pooled(df_cond, y_col='FFs', estimator='mean', granularity='neuron_level',split_by_age=True,  save=True)


if __name__ == "__main__":
    from scripts.utils.io import setup_logging
    setup_logging()

    main()