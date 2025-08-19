"""
pooled; slice_org
raw; mean-subtracted
all 7 metrics

#TODO: add stats results

"""
#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ibl_style.utils import MM_TO_INCH
from scripts.utils.plot_utils import figure_style
import figrid as fg
import ast

import config as C
from scripts.utils.plot_utils import create_slice_org_axes, map_p_value
from scripts.utils.io import read_table, save_figure

# def load_stats_results(df_path):
#     """
#     Load stats
#     """
#     return pd.read_csv(df_path)


def get_suffix(mean_subtraction):
    return 'meansub' if mean_subtraction else ''

# def get_vmin_vmax(metric):
#     ranges = {
#         'pre_fr': (0, 50), 'post_fr': (0, 50),
#         'fr_delta_modulation': (-10, 20),
#         'pre_ff': (0.5, 2.25), 'post_ff': (0.5, 2.25),
#         'ff_quench': (-0.8, 0.4), 'ff_quench_modulation': (-1.5, 2.5)
#     }
#     return ranges.get(metric, (-1, 1))


def get_vmin_vmax(metric):
    ranges = {
        'log_pre_fr': (0.5, 4.5), 'log_post_fr': (0.5, 4),
        'fr_delta_modulation': (-10, 20),
        'log_pre_ff': (-0.75, 0.75), 'log_post_ff': (-0.75, 0.75),
        'ff_quench': (-0.6, 0.4), 'ff_quench_modulation': (-1.5, 1.2)
    }
    return ranges.get(metric, (-1, 1))


def custom_fit_line(df, chain_table, mean_subtraction=False):

    age_range = np.linspace((df['mouse_age'].min())/365, (df['mouse_age'].max())/365, 100)
    logFF_preds = []

    intercept = chain_table.loc['mu', 'Mean']
    beta_age = chain_table.loc['age_years-age_years', 'Mean']
    if not mean_subtraction:
        beta_contrast = chain_table.loc['abs_contrast-abs_contrast', 'Mean']
        contrast_fixed = df['abs_contrast'].mean()

    beta_ntrials = chain_table.loc['n_trials-n_trials', 'Mean']
    cluster_region_effect = chain_table.loc['cluster_region-CA1', 'Mean']  # 举例

    # 
    ntrials_fixed = df['n_trials'].mean()
    cluster_region_fixed = 1  # 如果用的是 CA1；若baseline，则=0

    if not mean_subtraction:
        for age in age_range:
            pred = (
                intercept +
                beta_age * age +
                beta_contrast * contrast_fixed +
                beta_ntrials * ntrials_fixed +
                cluster_region_effect * cluster_region_fixed
            )
            logFF_preds.append(pred)
    else:
        for age in age_range:
            pred = (
                intercept +
                beta_age * age +
                beta_ntrials * ntrials_fixed +
                cluster_region_effect * cluster_region_fixed
            )
            logFF_preds.append(pred)
    logFF_preds = np.array(logFF_preds).T  # shape: [n_draws, 100]

    return age_range, logFF_preds


def custom_fit_line_region(df, chain_table, mean_subtraction=False):

    age_range = np.linspace((df['mouse_age'].min())/365, (df['mouse_age'].max())/365, 100)
    logFF_preds = []

    intercept = chain_table.loc['mu', 'Mean']
    beta_age = chain_table.loc['age_years', 'Mean']
    if not mean_subtraction:
        beta_contrast = chain_table.loc['abs_contrast', 'Mean']
        contrast_fixed = df['abs_contrast'].mean()

    beta_ntrials = chain_table.loc['n_trials', 'Mean']
    ntrials_fixed = df['n_trials'].mean()

    if not mean_subtraction:
        for age in age_range:
            pred = (
                intercept +
                beta_age * age +
                beta_contrast * contrast_fixed +
                beta_ntrials * ntrials_fixed 
            )
            logFF_preds.append(pred)
    else:
        for age in age_range:
            pred = (
                intercept +
                beta_age * age +
                beta_ntrials * ntrials_fixed 
            )
            logFF_preds.append(pred)
    logFF_preds = np.array(logFF_preds).T  # shape: [n_draws, 100]

    return age_range, logFF_preds


def plot_scatter_pooled(df, permut_df, BF_df, y_col='pre_fr', estimator='mean',
                                granularity='probe_level', ylim=(None, None),
                                save=True, mean_subtraction=False):

    figure_style()
    # fig, ax = plt.subplots(1, 1, figsize=(2.36, 2.36))  

    if mean_subtraction == False:
        if y_col in ['fr_delta_modulation', 'ff_quench_modulation']:
            df = df.drop(columns=['signed_contrast', 'abs_contrast'], errors='ignore')
            df = df.drop_duplicates(subset=['uuids', 'session_pid', 'mouse_age_months', 'age_group', y_col])

    if granularity == 'probe_level':
        df['number_neurons'] = df.groupby(['session_pid'])['uuids'].transform('nunique')
        agg_df = df.groupby(['session_pid','mouse_age','mouse_age_months', 'age_group','number_neurons'])[y_col].agg(estimator).reset_index()
        # df = agg_df 

    slope_age = permut_df['observed_val'].values[0]
    p_perm = permut_df['p_perm'].values[0]
    mapped_p_value = map_p_value(p_perm)

    BF_conclusion = BF_df.loc[BF_df['metric'] == y_col, 'BF10_age_category'].values[0]
    BF10 = BF_df.loc[BF_df['metric'] == y_col, 'BF10_age'].values[0]

    # sns.scatterplot(x='mouse_age_months', y=y_col, data=agg_df, hue='age_group',#size='num_datapoints',
    #     marker='.', legend=False, s=agg_df['number_neurons'],
    #     palette=palette, ax=ax)  # style_order=['M','F'],style='mouse_sex' #TODO: alpha=0.8


    if y_col in ['pre_fr','post_fr','pre_ff','post_ff']:
        # fig, ax = plt.subplots(1, 1, figsize=(3, 2.36))  
        fig, ax = plt.subplots(1, 1, figsize=(2.36, 2.36))  
        agg_df[f'log_{y_col}'] = np.log(agg_df[y_col]+ 1e-6)
        sns.scatterplot(x='mouse_age_months', y=f'log_{y_col}', data=agg_df, hue='age_group',#size='num_datapoints',
            marker='.', legend=False, s=agg_df['number_neurons'],
            palette=C.PALETTE, ax=ax)  # style_order=['M','F'],style='mouse_sex' #TODO: alpha=0.8
        if BF_conclusion == 'strong H1' or BF_conclusion == 'moderate H1':
            chain_table = pd.read_csv(os.path.join(C.RESULTSPATH, f'omnibus_{y_col}_BF_chain_table.csv'), index_col=0)
            age_range, logFF_preds = custom_fit_line(df, chain_table, mean_subtraction=mean_subtraction)
            sns.lineplot(x=age_range*12, y=logFF_preds, color='gray', lw=0.8, ax=ax)
        vmin, vmax = get_vmin_vmax(f'log_{y_col}')
        ax.set_ylim(vmin, vmax)

        # 4) 设置单轴（左轴）为 log 间距 + 原始单位标签
        if 'fr' in y_col:
            tick_vals = [2, 4, 8, 16, 32, 64]
        elif 'ff' in y_col:
            tick_vals = [0.5, 1, 2, 4]
            # tick_vals = [0.5, 0.8, 1.1, 1.7, 2.5]
        
        tick_pos = np.log(tick_vals)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels([str(v) for v in tick_vals])
        ax.set_ylabel(y_col)

        ax.set_ylim(np.log(min(tick_vals)), np.log(max(tick_vals)))

    else:
        fig, ax = plt.subplots(1, 1, figsize=(2.36, 2.36))  

        sns.scatterplot(x='mouse_age_months', y=y_col, data=agg_df, hue='age_group',#size='num_datapoints',
            marker='.', legend=False, s=agg_df['number_neurons'],
            palette=C.PALETTE, ax=ax)  # style_order=['M','F'],style='mouse_sex' #TODO: alpha=0.8
        
        if BF_conclusion == 'strong H1' or BF_conclusion == 'moderate H1':
            sns.regplot( x='mouse_age_months', y=y_col, data=agg_df,
                    scatter=False ,  color="gray", line_kws={"lw": 0.8}, ci=None, ax=ax)

        vmin, vmax = get_vmin_vmax(y_col)
        ax.set_ylim(vmin, vmax)

    if BF10 > 100:
        txt = fr" $\beta_{{\mathrm{{age}}}} = {slope_age:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} = {mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} > 100, $" + f" {BF_conclusion}"
    else:
        txt = fr" $\beta_{{\mathrm{{age}}}} = {slope_age:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} = {mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} = {BF10:.3f}, $" + f" {BF_conclusion}"


    ax.text(0.05, 1, txt, transform=ax.transAxes, fontsize=5, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    ax.set_xlabel('Age (months)')
    ax.set_ylabel(f'{y_col} ' if y_col in ['pre_fr', 'post_fr', 'pre_ff', 'post_ff'] else f'{y_col}')
    ax.set_title(f'Omnibus test: {y_col}')
    ax.set_xticks([5, 10, 15, 20])
    sns.despine(offset=2, trim=False, ax=ax)
    plt.tight_layout()

    if save:
        fname = f'Omnibus_{y_col}_{get_suffix(mean_subtraction)}_scatter.pdf'
        plt.savefig(os.path.join(C.FIGPATH, fname), dpi=500)

    plt.show()


def plot_scatter_by_region(df, permut_df, BF_df, y_col='pre_fr', estimator='mean',
                            granularity='probe_level', save=True, mean_subtraction=False):
    
    fig, axs = create_slice_org_axes(fg, MM_TO_INCH)
    figure_style()

    for region in C.ROIS:
        print(region)
        ax = axs[region]
        sub_df = df[df['cluster_region'] == region].copy()
        sub_BF_df = BF_df[BF_df['cluster_region'] == region]
        permut_df_region = permut_df[permut_df['cluster_region'] == region]
        
        # if sub_df.empty or stat.empty:
        if sub_df.empty:
            continue
        
        if not mean_subtraction:
            # These modulation metrics are per-neuron summary values, repeated across contrast levels → deduplicate
            if y_col in ['fr_delta_modulation', 'ff_quench_modulation']:
                sub_df = sub_df.drop(columns=['signed_contrast', 'abs_contrast'], errors='ignore')
                sub_df = sub_df.drop_duplicates(subset=['uuids', 'session_pid', 'mouse_age_months', 'age_group', y_col])

        if granularity == 'probe_level':
            # sub_df['number_neurons'] = sub_df.groupby(['session_pid'])['uuids'].transform('nunique')
            sub_df.loc[:, 'number_neurons'] = (sub_df.groupby('session_pid')['uuids'].transform('nunique'))
            agg_df = sub_df.groupby(['session_pid', 'mouse_age_months', 'age_group', 'number_neurons'])[y_col].agg(estimator).reset_index()
            agg_df = agg_df[['session_pid', 'mouse_age_months', y_col, 'age_group', 'number_neurons']].dropna()

        
        slope_age = permut_df_region['observed_val'].values[0]
        p_perm = permut_df_region['p_perm'].values[0]
        mapped_p_value = map_p_value(p_perm)

        BF_conclusion = sub_BF_df['BF10_age_category'].values[0]
        BF10 = sub_BF_df['BF10_age'].values[0]
        # BF_conclusion = sub_BF_df.loc[sub_BF_df['metric'] == y_col, 'BF10_age_category'].values[0]
        # BF10 = sub_BF_df.loc[sub_BF_df['metric'] == y_col, 'BF10_age'].values[0]

        if y_col in ['pre_fr', 'post_fr', 'pre_ff', 'post_ff']:
            agg_df[f'log_{y_col}'] = np.log(agg_df[y_col]+ 1e-6)

            sns.scatterplot(x='mouse_age_months', y=f'log_{y_col}', data=agg_df, hue='age_group',#size='num_datapoints',
                marker='.', legend=False, s=agg_df['number_neurons'],
                palette=C.PALETTE, ax=ax)  # style_order=['M','F'],style='mouse_sex' #TODO: alpha=0.8
            if BF_conclusion == 'strong H1' or BF_conclusion == 'moderate H1':
                #TODO:
                chain_table = read_table((C.RESULTSPATH / f'omnibus_{y_col}_{region}_BF_chain_table.csv'))
                # print(chain_table)
                age_range, logFF_preds = custom_fit_line_region(sub_df, chain_table, mean_subtraction=mean_subtraction)
                sns.lineplot(x=age_range*12, y=logFF_preds, color='gray', lw=0.8, ax=ax)
            vmin, vmax = get_vmin_vmax(f'log_{y_col}')
        else:
            sns.scatterplot(x='mouse_age_months', y=y_col, data=agg_df, hue='age_group',#size='num_datapoints',
                marker='.', legend=False, s=agg_df['number_neurons'],
                palette=C.PALETTE, ax=ax)  # style_order=['M','F'],style='mouse_sex' #TODO: alpha=0.8
            
            if BF_conclusion == 'strong H1' or BF_conclusion == 'moderate H1':
                sns.regplot( x='mouse_age_months', y=y_col, data=agg_df,
                        scatter=False , color="gray", line_kws={"lw": 0.8}, ci=None, ax=ax)

            vmin, vmax = get_vmin_vmax(y_col)
        # ax.set_ylim (-3, 15)
        ax.set_ylim(vmin, vmax)

        if BF10 > 100:
            txt = fr" $\beta_{{\mathrm{{age}}}} = {slope_age:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}}{mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} > 100, $" + f" {BF_conclusion}"
        else:
            txt = fr" $\beta_{{\mathrm{{age}}}} = {slope_age:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}}{mapped_p_value}$" +  f"\n$BF_{{\\mathrm{{10}}}} = {BF10:.3f}, $" + f" {BF_conclusion}"

        ax.text(0.05, 1.25, txt, transform=ax.transAxes, fontsize=4, verticalalignment='top', linespacing=0.8)
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
        filename = C.FIGPATH / f"supp_slice_org_{y_col}_{granularity}_{get_suffix(mean_subtraction)}_{estimator}_age_relationship_{C.ALIGN_EVENT}_2025.pdf"
        save_figure(fig, filename)

    plt.show()

# def main(mean_subtraction = False):

if __name__ == "__main__":
    mean_subtraction = True #TODO:
    # n_perm =1000

    if mean_subtraction:
        metrics_path = C.DATAPATH / "neural_metrics_summary_meansub.parquet"
        selected_metrics = C.METRICS_WITH_MEANSUB
    else:
        metrics_path = C.DATAPATH / "neural_metrics_summary_conditions.parquet"
        selected_metrics = C.METRICS_WITHOUT_MEANSUB

    print("Loading extracted neural metrics summary...")
    neural_metrics = read_table(metrics_path)
    neural_metrics['age_group'] = neural_metrics['mouse_age'].map(lambda x: 'old' if x > C.AGE_GROUP_THRESHOLD else 'young')
    neural_metrics['mouse_age_months'] = neural_metrics['mouse_age'] / 30

    for metric, est in selected_metrics:
        # if metric in ['fr_delta_modulation','ff_quench', 'ff_quench_modulation']:
        df_permut_path_pooled = C.RESULTSPATH / f"Omnibus_{metric}_{C.N_PERMUT_NEURAL_OMNIBUS}permutation_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_{'meansub' if mean_subtraction else ''}_2025.csv"
        df_permut_path_region = C.RESULTSPATH / f"Regional_{metric}_{C.N_PERMUT_NEURAL_REGIONAL}permutation_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_{'meansub' if mean_subtraction else ''}_2025.csv"
        df_BF_path_pooled = C.RESULTSPATH / f"Omnibus_{'meansub' if mean_subtraction else ''}BFs_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_2025_06Aug.csv"
        df_BF_path_region = C.RESULTSPATH / f"regional_{'meansub' if mean_subtraction else ''}BFs_{C.ALIGN_EVENT}_{metric}_{C.TRIAL_TYPE}_2025_06Aug.csv"

        # else:
        #     df_permut_path_pooled = C.DATAPATH / f"Omnibus_{metric}_{n_perm}permutation_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_{'meansub' if mean_subtraction else ''}_2025.csv"
        #     df_permut_path_region = C.DATAPATH / f"Regional_{metric}_{n_perm}permutation_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_{'meansub' if mean_subtraction else ''}_2025.csv"
        
        #     df_BF_path_pooled = C.DATAPATH / f"Omnibus_{'meansub' if mean_subtraction else ''}BFs_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_2025_06Aug.csv"
        #     df_BF_path_region = C.DATAPATH / f"regional_{'meansub' if mean_subtraction else ''}BFs_{C.ALIGN_EVENT}_{metric}_{C.TRIAL_TYPE}_2025_06Aug.csv"

        # print("Loading stats...")
        # #TODO: both permutation + BFs
        # pooled
        df_permut_pooled = read_table(df_permut_path_pooled)
        df_permut_region = read_table(df_permut_path_region)
        
        df_BF_pooled = read_table(df_BF_path_pooled)
        df_BF_pooled['BF10_age'] = df_BF_pooled['BF10_age'].apply(lambda x: ast.literal_eval(x)[0]).astype(float)
        df_BF_region = read_table(df_BF_path_region)
        df_BF_region['BF10_age'] = df_BF_region['BF10_age'].apply(lambda x: ast.literal_eval(x)[0]).astype(float)
        
        print(f"Plotting {metric}...")

        plot_scatter_pooled(neural_metrics, df_permut_pooled, df_BF_pooled, y_col=metric, estimator=est,
                            granularity='probe_level', save=True, mean_subtraction=mean_subtraction)
        plot_scatter_by_region(neural_metrics, df_permut_region, df_BF_region, y_col=metric, estimator=est,
                            granularity='probe_level', save=True, mean_subtraction=mean_subtraction)

# %%
