#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import figrid as fg
import seaborn as sns

from ibl_style.utils import MM_TO_INCH
from ibl_style.style import figure_style
from utils.config import datapath, figpath, align_event, trial_type, ROIs, palette, palette5, palette5_2groups, age_group_threshold, tolerance
from utils.plot_utils import create_slice_org_axes

def load_timecourse_data(df_path):
    """Load df_all_conditions containing FF and FR timecourses."""
    return pd.read_parquet(df_path)

def baseline_correct(df, y_col,estimator):
    """Subtract the value at time=0 from all timepoints for each unit/contrast group."""
    if estimator=='mean':
        baseline = df[df['timepoints'].abs() < tolerance].groupby(['uuids', 'abs_contrast'])[y_col].mean().rename('baseline')
    else:
        baseline = df[df['timepoints'].abs() < tolerance].groupby(['uuids', 'abs_contrast'])[y_col].median().rename('baseline')

    df = df.join(baseline, on=['uuids', 'abs_contrast'])
    df[y_col] = df[y_col] - df['baseline']
    return df.drop(columns='baseline')

def plot_modulation_time_courses_pooled(df, y_col='frs', estimator='median', granularity='neuron_level', save=True):
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
        palette=palette5, legend=False, ax=ax, errorbar = ('ci', 95),
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
        fname = f"Omnibus_group_modulation_{granularity}_{y_col}_{estimator}_timecourse_{align_event}_merged.pdf"
        fig.savefig(os.path.join(figpath, fname), dpi=300)

    plt.show()

def plot_modulation_time_courses_by_region(df, y_col='frs', estimator='median', granularity='neuron_level', split_by_age=False, save=True):
    """
    Plot regional baseline-corrected timecourses of neural metrics across absolute contrast levels.
    If `split_by_age` is True, plots each age group with separate contrast color maps.
    """
    fig, axs = create_slice_org_axes(fg, MM_TO_INCH)
    figure_style()

    for region in ROIs:
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
                cmap = palette5_2groups.get(group_name, palette5)
                sns.lineplot(
                    data=group_df, x='timepoints', y=y_col, hue='abs_contrast', ax=ax,
                    palette=cmap, lw=0.5, legend=False,
                    estimator=estimator, errorbar=None
                )
        else:
            sns.lineplot(
                data=sub_df, x='timepoints', y=y_col, estimator=estimator,lw=0.5,
                hue='abs_contrast', ax=ax, palette=palette5, legend=False,errorbar=None
            )

        ax.axvline(x=0, ls='--', lw=0.5, alpha=0.8, c='gray')
        ax.axvspan(-0.1, 0, facecolor='gray', alpha=0.1)
        ax.axvspan(0.16, 0.26, facecolor='gray', alpha=0.1)
        ax.set_xlim(-0.2, 0.8)

        sns.despine(offset=2, trim=False, ax=ax)
        if region in ['ACB', 'OLF', 'MBm', 'PO']:
            ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
        else:
            ax.set_xticks([])

        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(f'{granularity} {estimator} {y_col} (baseline corrected) time course', fontsize=8)
    if align_event == 'stim':
        fig.supxlabel('Time from stimulus onset (s)', fontsize=8).set_y(0.35)
    else:
        fig.supxlabel('Time from movement onset (s)', fontsize=8).set_y(0.35)

    if save:
        suffix = '2groups_' if split_by_age else ''
        fig.savefig(os.path.join(figpath, f"Slice_org_modulation_{suffix}{y_col}_timecourse_{align_event}_merged.pdf"), dpi=300)
    plt.show()

if __name__ == "__main__":
    df_cond_path = datapath / f"ibl_BWMLL_FFs_{align_event}_{trial_type}_conditions_2025_merged.parquet"
    print(f"Loading df_all_conditions from {df_cond_path} ...")

    df_cond = load_timecourse_data(df_cond_path)
    df_cond['age_group'] = df_cond['mouse_age'].map(lambda x: 'old' if x > age_group_threshold else 'young')
    df_cond['abs_contrast'] = df_cond['signed_contrast'].abs()/100

    print(f"Plotting omnibus frs contrast modulation time courses...")
    plot_modulation_time_courses_pooled(df_cond, y_col='frs', estimator='mean', granularity='neuron_level', save=True)

    print(f"Plotting omnibus FFs contrast modulation time courses...")
    plot_modulation_time_courses_pooled(df_cond, y_col='FFs', estimator='mean', granularity='neuron_stim_level', save=True)

    print(f"Plotting frs contrast modulation time courses by regions...")
    plot_modulation_time_courses_by_region(df_cond, y_col='frs', estimator='mean', granularity='neuron_level', split_by_age=False, save=True)

    print(f"Plotting FFs contrast modulation time courses by regions...")
    plot_modulation_time_courses_by_region(df_cond, y_col='FFs', estimator='mean', granularity='neuron_level', split_by_age=False, save=True)

    print(f"Plotting frs contrast modulation time courses by regions...[age group split]")
    plot_modulation_time_courses_by_region(df_cond, y_col='frs', estimator='mean', granularity='neuron_level', split_by_age=True, save=True)

    print(f"Plotting FFs contrast modulation time courses by regions...[age group split]")
    plot_modulation_time_courses_by_region(df_cond, y_col='FFs', estimator='mean', granularity='neuron_level', split_by_age=True, save=True)

# %% TODO: 为什么0条件那么曲折离奇？是因为estimator选错了还是其他什么原因？
df = df_cond.copy()
df['FFs'] = df.groupby(['uuids', 'timepoints', 'abs_contrast'])['FFs'].transform('median')
df = df.drop_duplicates(subset=['uuids', 'timepoints', 'abs_contrast', 'FFs', 'age_group'])

df = baseline_correct(df, 'FFs', 'median')