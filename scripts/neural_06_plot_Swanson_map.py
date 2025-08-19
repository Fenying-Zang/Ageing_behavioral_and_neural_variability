"""
plot swanson figs to show regional specificity

#TODO:note, the only difference lies in the stats file (_MSed)

"""
# neural_06_swanson_plot.py
#%%

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from iblatlas.plots import plot_swanson_vector
from iblatlas.atlas import BrainRegions
from scripts.utils.plot_utils import figure_style
import config as C

br = BrainRegions()

def region_table(beryl_names):
    ROI_df = pd.DataFrame({'beryl_name': beryl_names, 'ROI': beryl_names})

    ROI_mapping = {
        'VISa': 'PPC', 'VISam': 'PPC',
        'VISp': 'VISp+pm', 'VISpm': 'VISp+pm',
        'APN': 'MBm', 'MRN': 'MBm',
        'ACAv': 'ACA', 'ACAd': 'ACA',
        'PL': 'mPFC', 'ILA': 'mPFC',
        'ORBm': 'ORB', 'ORBl': 'ORB', 'ORBvl': 'ORB',
        'TTd': 'OLF', 'DP': 'OLF', 'AON': 'OLF',
        'LSr': 'LS', 'LSc': 'LS', 'LSv': 'LS'
    }

    ROI_df['ROI'] = ROI_df['beryl_name'].map(ROI_mapping).fillna(ROI_df['beryl_name'])
    ROI_df['cosmos_name'] = ROI_df['beryl_name'].apply(lambda r: br.acronym2acronym(r, mapping='Cosmos')[0])
    ROI_df['swanson_name'] = ROI_df['beryl_name'].apply(lambda r: br.acronym2acronym(r, mapping='Swanson')[0])

    return ROI_df


def load_stats_results(y_col, mean_subtraction=False, n_permut=C.N_PERMUT_NEURAL_REGIONAL):
    suffix = 'meansub' if mean_subtraction else ''
    if y_col in ['ff_quench','fr_delta_modulation', 'ff_quench_modulation']:
        fname = f'Regional_{y_col}_{n_permut}permutation_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_{suffix}_2025.csv'
    else:
        fname = f'Regional_{y_col}_{n_permut}permutation_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_{suffix}_2025.csv'
    return pd.read_csv(C.RESULTSPATH / fname)

def get_vmin_vmax(metric):
    ranges = {
        'pre_fr': (-1, 1), 'post_fr': (-1, 1),
        'fr_delta_modulation': (-4, 4),
        'pre_ff': (-0.4, 0.4), 'post_ff': (-0.4, 0.4),
        'ff_quench': (-0.4, 0.4), 'ff_quench_modulation': (-0.6, 0.6)
    }
    return ranges.get(metric, (-1, 1))


def plot_swanson(metric, ROI_df, stats_df, mean_subtraction=False):
    cmap = LinearSegmentedColormap.from_list(
        "swanson_cmap",
        C.COLORS_SWANSON_INVERT if 'ff_quench' in metric else C.COLORS_SWANSON,
        N=256
    )

    merged_df = ROI_df.merge(
        stats_df[['cluster_region', 'observed_val', 'p_perm']],
        left_on='ROI', right_on='cluster_region', how='left'
    )

    vmin, vmax = get_vmin_vmax(metric)
    figure_style()
    fig, ax = plt.subplots(figsize=(4.72, 2.36))

    plot_swanson_vector(
        merged_df['beryl_name'], merged_df['observed_val'],
        cmap=cmap, vmin=vmin, vmax=vmax, br=br,
        empty_color='silver', show_cbar=True, annotate=True,
        annotate_list=merged_df['swanson_name'], fontsize=2, ax=ax
    )

    # sig_regions = merged_df.loc[merged_df['p_perm'] <= 0.01, 'ROI'].dropna().unique().tolist()
    # print(f'Significant regions for {metric}: {sig_regions}')
    # ax.text(0.1, -0.3, f'{sig_regions}', fontsize=5, transform=ax.transAxes)
    ax.set_title(metric, fontsize=8)
    ax.set_axis_off()

    fname = f"Swanson_{'meansub_' if mean_subtraction else ''}{metric}.pdf"
    fig.savefig(os.path.join(C.FIGPATH, fname), dpi=300)

def main(mean_subtraction=False):
    if mean_subtraction:
        selected_metrics = C.METRICS_WITH_MEANSUB
    else:
        selected_metrics = C.METRICS_WITHOUT_MEANSUB
    ROI_df = region_table(C.BERYL_NAMES)
    # metrics = ['pre_fr', 'post_fr', 'fr_delta_modulation', 'pre_ff', 'post_ff', 'ff_quench', 'ff_quench_modulation']

    for metric, _ in selected_metrics:
        print(f"Plotting Swanson for {metric}...")
        stats_df = load_stats_results(metric, mean_subtraction=mean_subtraction) 
        plot_swanson(metric, ROI_df, stats_df, mean_subtraction=mean_subtraction)
        # print('range of values:', stats_df['observed_val'].min(), stats_df['observed_val'].max(), stats_df['observed_val'].mean())


if __name__ == "__main__":

    main(mean_subtraction=False)


