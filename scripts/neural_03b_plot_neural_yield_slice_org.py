# neural_04_plot_neural_yield.py
#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ibl_style.style import figure_style
from ibl_style.utils import MM_TO_INCH
from utils.config import datapath, figpath, align_event, trial_type, ROIs, palette
from utils.plot_utils import create_slice_org_axes
import figrid as fg

def load_neural_yield_table():
    table = pd.read_parquet(
        os.path.join(datapath, f'ibl_BWMLL_neural_yield_{align_event}_{trial_type}_2025_full.parquet'))
    table['neural_yield'] = table['n_cluster'] / table['n_channel']
    table['age_group'] = table['age_at_recording'].map(lambda x: 'old' if x > 300 else 'young')
    table['age_years'] = table['age_at_recording'] / 365
    return table

def plot_yield_by_region(df, stats_df, y_var='n_cluster', save=True):
    result_df = stats_df[stats_df['y_var'] == y_var]
    fig, axs = create_slice_org_axes(fg, MM_TO_INCH)
    figure_style()

    for region in ROIs:
        ax = axs[region]
        sub_df = df[df['Beryl_merge'] == region]
        stat = result_df[result_df['cluster_region'] == region]

        if sub_df.empty or stat.empty:
            continue

        beta = stat['observed_val'].values[0]
        p_perm = stat['p_perm'].values[0]
        # p_fdr = stat['p_corrected'].values[0]

        # txt = fr"$\beta_{{\mathrm{{age}}}} = {beta:.3f}$\n$p_{{\mathrm{{perm}}}} = {p_perm:.3f}$\n$p_{{\mathrm{{fdr}}}} = {p_fdr:.3f}$"
        txt = fr"$\beta_{{\mathrm{{age}}}} = {beta:.3f}$\n$p_{{\mathrm{{perm}}}} = {p_perm:.3f}$"

        ax.text(0.05, 1.2, txt, transform=ax.transAxes, fontsize=4, verticalalignment='top')

        if p_perm < 0.01:
            sns.regplot(data=sub_df, x=sub_df['age_at_recording'] / 30, y=y_var,
                        marker='.', color="1", line_kws=dict(color="gray"), ax=ax)

        sns.scatterplot(data=sub_df, x=sub_df['age_at_recording'] / 30, y=y_var, hue='age_group',
                        palette=palette, ax=ax, legend=False, marker='.')

        ax.set_ylabel(region)
        ax.set_xlabel("")
        ax.set_xticks([5, 10, 15, 20] if region in ['ACB', 'OLF', 'MBm', 'PO'] else [])
        ax.set_ylim(0, 180 if y_var == 'n_cluster' else 1.8)
        sns.despine(offset=2, trim=False, ax=ax)

    fig.suptitle(f"{y_var} across regions", fontsize=8)
    fig.supxlabel("Age (months)", fontsize=8).set_y(0.35)

    if save:
        fname = f"F2S1_slice_org_{y_var}_{align_event}_permutation_2025.pdf"
        fig.savefig(os.path.join(figpath, fname))
        print(f"Saved figure to {figpath}/{fname}")

    plt.show()


if __name__ == "__main__":
    
    print("Loading data...")
    df = load_neural_yield_table()
    stats = pd.read_csv(os.path.join(
        datapath, f'Neural_yield_ols_permut_1000permutation_{align_event}_{trial_type}_2025.csv'))

    for y_var in ['n_cluster', 'neural_yield']:
        plot_yield_by_region(df, stats, y_var)

