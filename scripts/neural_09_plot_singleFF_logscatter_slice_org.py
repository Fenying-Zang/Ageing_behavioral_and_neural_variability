"""

plot singleFF (var vs mean) scatter plot
Figure 4 S1. The relationship between spike count variance and spike count mean.

"""
import config as C
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator
import figrid as fg
from ibl_style.utils import get_coords, MM_TO_INCH, double_column_fig
from scripts.utils.plot_utils import figure_style
from scripts.utils.io import read_table,save_figure
from scripts.utils.io import read_table, save_figure, setup_logging
import logging


setup_logging()
log = logging.getLogger(__name__)


def create_slice_org_axes_singleFF(fg, MM_TO_INCH, fig=None):

    if fig is None:
        fig = double_column_fig()

    width, height = fig.get_size_inches() / MM_TO_INCH #180, 170
    xspans = get_coords(width, ratios=[1, 1, 1, 1], space=20, pad=5, span=(0, 1))#from 0-1
    panel_width = (xspans[0][1] - xspans[0][0]) * width  # 单位：mm

    num_rows = 5
    space = 10  # mm
    pad = 5     # mm
    panel_height = panel_width
    total_height = num_rows * panel_height + (num_rows - 1) * space + 2 * pad
    fig.set_size_inches(width * MM_TO_INCH, total_height * MM_TO_INCH)
    yspans = get_coords(total_height, ratios=[1]*num_rows, space=space, pad=pad, span=(0, 1))

    layout = {
        'MOs': (0, 0), 'ACA': (0, 1), 'CP': (0, 2), 'LS': (0, 3), 'ACB': (0, 4),
        'mPFC': (1, 2), 'ORB': (1, 3), 'OLF': (1, 4),
        'VISp+pm': (2, 2), 'SCm': (2, 3), 'MBm': (2, 4),
        'PPC': (3, 0), 'CA1': (3, 1), 'DG': (3, 2), 'LP': (3, 3), 'PO': (3, 4),
    }

    axs = {region: fg.place_axes_on_grid(fig, xspan=xspans[xi], yspan=yspans[yi])
        for region, (xi, yi) in layout.items()}
    
    return fig, axs

def plot_singleFF_mean_var(df, timepoint, save=True):

    df_timepoint = df[np.abs(df['timepoints'] - timepoint) < C.TOLERANCE]
    fig, axs = create_slice_org_axes_singleFF(fg, MM_TO_INCH)
    figure_style()

    for region in C.ROIS:
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
            marker='.', palette=C.PALETTE, ax=ax,
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

                # xs = np.logspace(0.01, 1000, 1000)
                if slope < 1:
                    ax.plot([0.01, 100], [0.01, 100*slope], color=C.PALETTE[group], linewidth=0.5)
                else:
                    ax.plot([0.01, 100/slope], [0.01, 100], color=C.PALETTE[group], linewidth=0.5)
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

    fig.supxlabel('Spike count mean (spikes)', font="Arial", fontsize=8).set_y(0.05)
    fig.supylabel('Spike count variance (spikes^2)', font="Arial", fontsize=8)

    if save:
        if timepoint==C.PRE_TIME:
            save_figure(fig, C.FIGPATH / f"single_FF_logscatter_pre-stim_{C.ALIGN_EVENT}.pdf", add_timestamp=True)
        elif timepoint==C.POST_TIME:
            save_figure(fig, C.FIGPATH / f"single_FF_logscatter_post-stim_{C.ALIGN_EVENT}.pdf", add_timestamp=True)


def main():
    
    print("Loading df_all_conditions...")
    df_cond_path = C.DATAPATH / f"ibl_BWMLL_FFs_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_conditions_2025.parquet"
    df_cond = read_table(df_cond_path)
    df_cond['age_group'] = df_cond['mouse_age'].map(lambda x: 'old' if x > C.AGE_GROUP_THRESHOLD else 'young')
    
    print("Plotting FF mean vs FF var...")
    plot_singleFF_mean_var(df_cond, C.PRE_TIME, save=True)
    plot_singleFF_mean_var(df_cond, C.POST_TIME, save=True)


if __name__ == "__main__":
    from scripts.utils.io import setup_logging
    setup_logging()
    
    main()


