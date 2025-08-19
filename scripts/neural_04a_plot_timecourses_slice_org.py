"""
frs time courses  
FF time courses
"""
#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import figrid as fg
import seaborn as sns
from ibl_style.utils import MM_TO_INCH
from scripts.utils.plot_utils import figure_style
import config as C
from scripts.utils.plot_utils import create_slice_org_axes
from scripts.utils.io import read_table


def load_timecourse_data(df_path):
    """
    Load df_all_conditions containing FF and FR timecourses.
    """
    return read_table(df_path)


def get_suffix(mean_subtraction):
    return 'meansub' if mean_subtraction else ''


def plot_time_courses_pooled(df, y_col='frs', estimator='mean',
                                granularity='neuron_level',errorbar = ('ci', 95),
                                save=True, mean_subtraction=False):
    
    if granularity == 'neuron_level' and not mean_subtraction:
        # Collapse across contrast levels per neuron
        df[y_col] = df.groupby(['uuids','timepoints'])[y_col].transform(estimator)
        df = df.drop_duplicates(subset=['uuids','timepoints',y_col,'age_group'])

    num_datapoints = df.groupby(['age_group'])['uuids'].nunique()

    figure_style()
    fig, ax = plt.subplots(figsize=(3, 2))

    sns.lineplot(
        data=df, x='timepoints', y=y_col,
        estimator=estimator, 
        hue='age_group', palette=C.PALETTE, hue_order=['young','old'],
       errorbar=errorbar, legend=False, ax=ax 
    )

    ax.text(0.8, 0.78, f'{num_datapoints.get("old", 0)} old', transform=ax.transAxes,
            font="Arial", fontsize=6, c=C.PALETTE['old'])
    ax.text(0.8, 0.95, f'{num_datapoints.get("young", 0)} young', transform=ax.transAxes,
            font="Arial", fontsize=6, c=C.PALETTE['young'])

    ax.axvline(x=0,  lw=0.5, alpha=0.8, c='gray')
    ax.axvspan(-0.1, 0, facecolor='gray', alpha=0.1)
    ax.axvspan(0.16, 0.26, facecolor='gray', alpha=0.1)

    ax.set_xlim(-0.2, 0.8)
    ax.set_xlabel("Time from stimulus onset (s)", font="Arial", fontsize=8)
    ax.set_ylabel(f'{estimator} {y_col} ', font="Arial", fontsize=8)
    ax.set_title(f"{granularity}, {y_col}; {errorbar}", font="Arial", fontsize=8)

    sns.despine(offset=2, trim=False, ax=ax)
    plt.tight_layout()

    if save:
        fname = f"Omnibus_group_{granularity}_{get_suffix(mean_subtraction)}_{y_col}_{estimator}_timecourse_{C.ALIGN_EVENT}.pdf"
        fig.savefig(os.path.join(C.FIGPATH, fname), dpi=300)

    plt.show()


def plot_time_courses_by_region(df, y_col='frs', estimator='mean',errorbar = ('ci', 95),
                                granularity='neuron_level', save=True, mean_subtraction=False):
    """Plot firing rate or FF timecourses organized by brain region."""
    fig, axs = create_slice_org_axes(fg, MM_TO_INCH)
    figure_style()

    for region in C.ROIS:
        ax = axs[region]
        sub_df = df[df['cluster_region'] == region]
        if sub_df.empty:
            continue

        # Collapse across contrast levels per neuron if using neuron-level granularity
        if granularity == 'neuron_level' and not mean_subtraction:
            agg_df = sub_df.groupby(['uuids', 'age_group', 'timepoints'])[y_col].agg(estimator).reset_index()
            sub_df = agg_df  # 只保留 collapsed 后的 summary 数据

        # Count datapoints at time zero
        mydf_temp = sub_df[np.abs(sub_df['timepoints']) < C.TOLERANCE]
        num_datapoints = mydf_temp.groupby('age_group')['uuids'].nunique()

        # Plot timecourse
        sns.lineplot(data=sub_df, x='timepoints', y=y_col, estimator=estimator,
                     hue='age_group', hue_order=['young', 'old'], ax=ax,
                     palette=C.PALETTE, errorbar=errorbar, legend=False)

        # Annotate sample size
        if 'old' in num_datapoints:
            ax.text(0.7, 0.84, f"{num_datapoints['old']}", transform=ax.transAxes,
                    fontsize=6, color=C.PALETTE['old'])
        if 'young' in num_datapoints:
            ax.text(0.7, 1.0, f"{num_datapoints['young']}", transform=ax.transAxes,
                    fontsize=6, color=C.PALETTE['young'])

        # Visual aids
        ax.axvline(x=0, lw=0.5, alpha=0.8, c='gray')
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

    # Final figure settings
    fig.suptitle(f'{granularity} {estimator} {y_col} time course', fontsize=8)
    if C.ALIGN_EVENT == 'stim':
        fig.supxlabel('Time from stimulus onset (s)', fontsize=8).set_y(0.35)
    else:
        fig.supxlabel('Time from movement onset (s)', fontsize=8).set_y(0.35)

    if save:
        fig.savefig(os.path.join(C.FIGPATH, f"Slice_org_{get_suffix(mean_subtraction)}_{granularity}_{y_col}_{estimator}_timecourse_{C.ALIGN_EVENT}.pdf"), dpi=300)
    plt.show()


def main(mean_subtraction=True):
    if mean_subtraction:
        y_FF_col = 'FFs_residuals'
        y_fr_col = 'frs_residuals'
        df_cond_path = C.DATAPATH / f"ibl_BWMLL_FFs_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_2025.parquet"
    else:
        y_FF_col = 'FFs'
        y_fr_col = 'frs'
        df_cond_path = C.DATAPATH / f"ibl_BWMLL_FFs_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_conditions_2025.parquet"

    print(f"Loading {get_suffix(mean_subtraction)} df_all_conditions...")
    df_cond = load_timecourse_data(df_cond_path)
    df_cond['age_group'] = df_cond['mouse_age'].map(lambda x: 'old' if x > C.AGE_GROUP_THRESHOLD else 'young')

    print(f"Plotting {get_suffix(mean_subtraction)} pooled frs time courses...")
    plot_time_courses_pooled(df_cond, y_col=y_fr_col, estimator='mean',
                                    granularity='neuron_level', errorbar=('ci', 95),
                                    save=True, mean_subtraction=mean_subtraction)

    print(f"Plotting {get_suffix(mean_subtraction)} frs time courses...")
    plot_time_courses_by_region(df_cond, y_col=y_fr_col, estimator='mean',
                                granularity='neuron_level', errorbar=('ci', 95), 
                                save=True, mean_subtraction=mean_subtraction)
    
    print(f"Plotting {get_suffix(mean_subtraction)} pooled FFs time courses...")
    plot_time_courses_pooled(df_cond, y_col=y_FF_col, estimator='mean',
                                    granularity='neuron_level', errorbar=('ci', 95),
                                    save=True, mean_subtraction=mean_subtraction)

    print(f"Plotting {get_suffix(mean_subtraction)} FFs time courses...")
    plot_time_courses_by_region(df_cond, y_col=y_FF_col, estimator='mean',
                                granularity='neuron_level', errorbar=('ci', 95), 
                                save=True, mean_subtraction=mean_subtraction)


if __name__ == "__main__":

    main(mean_subtraction=True)
