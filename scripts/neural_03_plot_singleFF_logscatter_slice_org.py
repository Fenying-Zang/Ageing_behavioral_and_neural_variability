"""

plot singleFF (var vs mean) scatter plot
neural_07_singleFF_logscatter.py

"""
#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator
import figrid as fg
from ibl_style.utils import get_coords, MM_TO_INCH, double_column_fig
from ibl_style.style import figure_style

from utils.config import tolerance, PRE_TIME, POST_TIME, ROIs, palette, datapath,figpath, align_event, trial_type,age_group_threshold


def create_slice_org_axes_singleFF(fg, MM_TO_INCH, fig=None):

    if fig is None:
        fig = double_column_fig()

    # Get the dimensions of the figure in mm
    width, height = fig.get_size_inches() / MM_TO_INCH #180, 170

    xspans = get_coords(width, ratios=[1, 1, 1, 1], space=20, pad=5, span=(0, 1))#from 0-1

    # Step 1: 子图宽度来自xspans（以xspans[0]为例）
    panel_width = (xspans[0][1] - xspans[0][0]) * width  # 单位：mm

    # Step 2: 用这个宽度来设置每个子图的高度，从而推算图总高
    num_rows = 5
    space = 10  # mm
    pad = 5     # mm
    panel_height = panel_width
    total_height = num_rows * panel_height + (num_rows - 1) * space + 2 * pad

    # Step 3: 更新figure尺寸（宽度不变，高度更新）
    fig.set_size_inches(width * MM_TO_INCH, total_height * MM_TO_INCH)

    # Step 4: 重新计算yspans（注意这里的total_height要传进去）
    yspans = get_coords(total_height, ratios=[1]*num_rows, space=space, pad=pad, span=(0, 1))

    layout = {
        'MOs': (0, 0), 'ACA': (0, 1), 'CP': (0, 2), 'LS': (0, 3), 'ACB': (0, 4),
        'mPFC': (1, 2), 'ORB': (1, 3), 'OLF': (1, 4),
        'VISp+pm': (2, 2), 'SCm': (2, 3), 'MBm': (2, 4),
        'PPC': (3, 0), 'CA1': (3, 1), 'DG': (3, 2), 'LP': (3, 3), 'PO': (3, 4),
    }

    axs = {
        region: fg.place_axes_on_grid(fig, xspan=xspans[xi], yspan=yspans[yi])
        for region, (xi, yi) in layout.items()
    }

    return fig, axs

def load_timecourse_data(df_path):
    return pd.read_parquet(df_path)


def plot_singleFF_mean_var (df, timepoint, save=True):

    df_timepoint = df[np.abs(df['timepoints'] - timepoint) < tolerance]
    fig, axs = create_slice_org_axes_singleFF(fg, MM_TO_INCH)
    figure_style()

    for region in ROIs:
        ax = axs[region]
        sub_df = df_timepoint[df_timepoint['cluster_region'] == region]
        if sub_df.empty:
            continue
        
        # Log scale
        ax.set_yscale('log')
        ax.set_xscale('log')
        
        # Scatter plot
        sns.scatterplot(
            data=sub_df, x="FF_mean", y="FF_variance",
            hue="age_group", hue_order=['young', 'old'],
            marker='.', palette=palette, ax=ax,
            s=0.7, edgecolor='none', linewidth=0, legend=False
        )

        # Plot fitted line for each group
        for idx_g, group in enumerate(['young', 'old']):
            group_df = sub_df[sub_df['age_group'] == group]
            x = group_df['FF_mean'].values
            y = group_df['FF_variance'].values

            if len(x) > 1:
                slope, _, _, _ = np.linalg.lstsq(x[:, np.newaxis], y, rcond=None)
                slope = slope[0]
                
                # Draw line (manual, from (0.01, 0.01*slope) to (100, 100*slope))
                # xs = np.array([0.01, 100])
                # ys = xs * slope
                # ax.plot(xs, ys, color=palette[group], linewidth=0.5)

                xs = np.logspace(0.01, 1000, 1000)
                if slope<1:
                    ax.plot([0.01, 100], [0.01, 100*slope], color=palette[group],linewidth=0.5)
                else:
                    ax.plot([0.01, 100/slope], [0.01, 100], color=palette[group],linewidth=0.5)
        # Add FF=1 line
        lims = [0.01, 1000]
        ax.plot(lims, lims, linestyle='--', color='gray', linewidth=0.5)
        # ax.plot([min(ax.get_xlim()), max(ax.get_xlim())], [min(ax.get_xlim()), max(ax.get_xlim())], linestyle='--', color='gray')

        # Log ticks and limits
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=100))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=100))

        # Axis ticks conditionally shown
        if region in ['ACB', 'OLF', 'MBm', 'PO']:
            ax.set_xticks([0.01, 0.1, 1.0, 10, 100, 1000])
        else:
            ax.set_xticks([])
        if region in ['MOs','ACA','CP','LS','ACB']:
            ax.set_yticks([0.01, 0.1, 1.0, 10, 100, 1000])
        else:
            ax.set_yticks([])

        ax.set_xlabel(" ")
        ax.set_ylabel(" ")  
        sns.despine(offset=2, trim=False, ax=ax)

    fig.supxlabel('FF mean', font="Arial", fontsize=8).set_y(0.05)
    fig.supylabel('FF variance', font="Arial", fontsize=8)
    # plt.tight_layout() #TODO:新加的，看情况是否要去掉
    # plt.tight_layout()  # optional: may interfere with fig.set_size_inches layout

    if save:
        fig.savefig(
            os.path.join(figpath, f"single_FF_logscatter_{timepoint}_{align_event}.pdf"),
            dpi=300
        )

    plt.show()


if __name__ == "__main__":

    df_cond_path = datapath / f"ibl_BWMLL_FFs_{align_event}_{trial_type}_conditions_2025_merged.parquet"

    print("Loading df_all_conditions...")
    df_cond = load_timecourse_data(df_cond_path)
    df_cond['age_group'] = df_cond['mouse_age'].map(lambda x: 'old' if x > age_group_threshold else 'young')
    
    print("Plotting FF mean vs FF var...")
    plot_singleFF_mean_var (df_cond, PRE_TIME, save=True)
    plot_singleFF_mean_var (df_cond, POST_TIME, save=True)
# %%
