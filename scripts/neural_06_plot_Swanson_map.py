"""
plot swanson figs to show regional specificity

#TODO:note, the only difference lies in the stats file (_MSed)

"""
# neural_06_swanson_plot.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from iblatlas.plots import plot_swanson_vector
from iblatlas.atlas import BrainRegions
from ibl_style.style import figure_style

from utils.config import datapath, figpath, align_event, trial_type, beryl_names
from utils.config import colors_swanson, colors_swanson_invert

# Global brain region object
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


def load_stats_results(y_col, mean_subtraction=False, n_permut=2000):
    suffix = 'MSed_' if mean_subtraction else ''
    fname = f'regional_{y_col}_{n_permut}permutation_{align_event}_{trial_type}_{suffix}2025.csv'
    return pd.read_csv(os.path.join(datapath, fname))

def get_vmin_vmax(metric):
    ranges = {
        'pre_fr': (-0.5, 0.5), 'post_fr': (-0.5, 0.5),
        'fr_delta_modulation': (-2, 2),
        'pre_ff': (-0.3, 0.3), 'post_ff': (-0.3, 0.3),
        'ff_quench': (-0.3, 0.3), 'ff_quench_modulation': (-0.3, 0.3)
    }
    return ranges.get(metric, (-1, 1))


def plot_swanson(metric, ROI_df, stats_df, mean_subtraction=False):
    cmap = LinearSegmentedColormap.from_list(
        "swanson_cmap",
        colors_swanson_invert if 'modulation' in metric else colors_swanson,
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

    sig_regions = merged_df.loc[merged_df['p_perm'] <= 0.01, 'ROI'].dropna().unique().tolist()
    print(f'Significant regions for {metric}: {sig_regions}')
    ax.text(0.1, -0.3, f'{sig_regions}', fontsize=5, transform=ax.transAxes)
    ax.set_title(metric, fontsize=8)
    ax.set_axis_off()

    fname = f"{metric}_swanson_panels_{'MSed_' if mean_subtraction else ''}{align_event}.pdf"
    fig.savefig(os.path.join(figpath, fname), dpi=300)

if __name__ == "__main__":

    ROI_df = region_table(beryl_names)
    metrics = ['pre_fr', 'post_fr', 'fr_delta_modulation', 'pre_ff', 'post_ff', 'ff_quench', 'ff_quench_modulation']

    for metric in metrics:
        print(f"Plotting Swanson for {metric}...")
        stats_df = load_stats_results(metric, mean_subtraction=False) #TODO
        plot_swanson(metric, ROI_df, stats_df, mean_subtraction=False)



