"""
permutation on neural yield

"""
#%%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.utils.plot_utils import figure_style
from ibl_style.utils import MM_TO_INCH
import config as C
from scripts.utils.plot_utils import create_slice_org_axes, map_p_value
import figrid as fg
from scripts.utils.data_utils import shuffle_labels_perm, bf_gaussian_via_pearson, interpret_bayes_factor
from joblib import Parallel, delayed
from statsmodels.genmod.families import Gaussian
from statsmodels.formula.api import glm
from tqdm import tqdm
from scripts.utils.plot_utils import plot_permut_test
from scripts.utils.io import read_table

def load_neural_yield_table():
    table = read_table(
        # os.path.join(C.DATAPATH, f'ibl_BWMLL_neural_yield_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_2025_merged.parquet'))
        os.path.join(C.DATAPATH, f'ibl_BWMLL_neural_yield_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_2025.parquet'))

    table['neural_yield'] = table['n_cluster'] / table['n_channel']
    table['age_group'] = table['age_at_recording'].map(lambda x: 'old' if x > C.AGE_GROUP_THRESHOLD else 'young')
    table['age_years'] = table['age_at_recording'] / 365
    return table

def single_permutation(i, data, permuted_label, formula2use, family_func):
    try:
        shuffled_data = data.copy()
        shuffled_data['age_years'] = permuted_label

        model = glm(formula=formula2use, data=shuffled_data, family=family_func).fit()
        
        return model.params["age_years"]
    except Exception as e:
        print(f"Permutation {i} failed: {e}")
        return np.nan

def run_permutation_test(data, this_age, formula2use, family_func, n_permut, n_jobs):

    permuted_labels, _ = shuffle_labels_perm(
        labels1=this_age,
        labels2=None,
        shuffling='labels1_global',
        n_permut=n_permut,
        random_state=123,
        n_cores=n_jobs
    )

    null_dist = Parallel(n_jobs=n_jobs)(
        delayed(single_permutation)(i, data, permuted_labels[i], formula2use, family_func)
        for i in tqdm(range(n_permut))
    )

    null_dist = np.array(null_dist)
    valid_null = null_dist[~np.isnan(null_dist)]

    model_obs = glm(formula=formula2use, data=data, family=family_func).fit()
    observed_val = model_obs.params["age_years"]
    observed_val_p = model_obs.pvalues["age_years"]
    p_perm = (np.sum(np.abs(valid_null) >= np.abs(observed_val)) + 1) / (len(valid_null) + 1)

    return observed_val, observed_val_p, p_perm, valid_null

def get_permut_results (y_var, age2use, neural_yield_table):
    filename = C.RESULTSPATH / f"regional_{y_var}_{age2use}_{C.N_PERMUT_NEURAL_REGIONAL}permutation_NEW.csv"
    if filename.exists():
        permut_df = pd.read_csv(filename)
    else:

        # shuffling='labels1_global'#'labels1_based_on_2'
        family_func = Gaussian()
        formula2use = f"{y_var} ~ age_years"
        region_results = []
        for region in C.ROIS:
            print(f'Processing region {region}')
            region_data = neural_yield_table[neural_yield_table['Beryl_merge']==region] 

            region_data = region_data[~ np.isnan(region_data[y_var])].reset_index(drop=True)
            this_data = region_data[y_var].values
            this_age = region_data['age_years'].values
            observed_val, observed_val_p, p_perm, valid_null = run_permutation_test(
                data=region_data,
                this_age=this_age,
                formula2use=formula2use,
                family_func=family_func,
                n_permut=C.N_PERMUT_NEURAL_REGIONAL,
                n_jobs=6
            )
            print(f"results for {y_var}: \n  beta = {observed_val:.4f}, p_perm = {p_perm:.4f}")
            # if plot_permt_result:
            #     plot_permut_test(null_dist=valid_null, observed_val=observed_val, p=p_perm, mark_p=None, metric=y_var, save_path=C.FIGPATH, show=True, region=region)

            region_results.append({
                'cluster_region': region,
                'metric': y_var,
                'n_perm': C.N_PERMUT_NEURAL_REGIONAL,
                'formula': formula2use,
                'observed_val': observed_val,
                'observed_val_p': observed_val_p,
                'p_perm': p_perm,
                'ave_null_dist': valid_null.mean()
            }) #  'null_dist': valid_nul

        permut_df = pd.DataFrame(region_results)
        permut_df.to_csv(filename, index=False) 
    return permut_df


def get_bf_results(content, age2use, df):
    filename = C.RESULTSPATH / f"regional_beyesfactor_{content}_NEW.csv"
    if filename.exists():
        BF_df = pd.read_csv(filename)
        # BF10 = BF_dict['BF10'].values[0]
        # BF_conclusion = BF_dict['BF_conclusion'].values[0]
    else:
        region_results = []
        for region in C.ROIS:
            print(f'Processing region {region}')
            region_data = df[df['Beryl_merge']==region] 

            region_data = region_data[~ np.isnan(region_data[content])].reset_index(drop=True)
            # this_data = region_data[content].values
            # this_age = region_data[age2use].values
            BF_dict = bf_gaussian_via_pearson(region_data, content, age2use)
            # print(f"BF10 for {content} vs. {age2use}: {BF_dict['BF10']:.3f}, r={BF_dict['r']:.3f}, n={BF_dict['n']}")
            BF10 = BF_dict['BF10']
            BF_conclusion = interpret_bayes_factor(BF10)

            region_results.append({
                'metric': content,
                'cluster_region': region,
                'BF10': BF10,
                'BF_conclusion': BF_conclusion
            })
        BF_df = pd.DataFrame(region_results)
        BF_df.to_csv(filename, index=False)

    return BF_df


def plot_yield_by_region(df, permut_df, bf_df, y_var='n_cluster', save_fig=True):

    fig, axs = create_slice_org_axes(fg, MM_TO_INCH)
    figure_style()

    for region in C.ROIS:
        ax = axs[region]
        sub_df = df[df['Beryl_merge'] == region]
        region_permut_df = permut_df[permut_df['cluster_region'] == region]
        result_bf_df = bf_df[bf_df['cluster_region'] == region]
        
        if sub_df.empty or region_permut_df.empty or result_bf_df.empty:
            continue

        beta = region_permut_df['observed_val'].values[0]
        p_perm = region_permut_df['p_perm'].values[0]
        p_perm_mapped = map_p_value(p_perm)
        BF10 = result_bf_df['BF10'].values[0]
        BF_conclusion = result_bf_df['BF_conclusion'].values[0]

        if BF10 > 100:
            txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} {p_perm_mapped}$" +  f"\n$BF_{{\\mathrm{{10}}}} > 100, $" + f" {BF_conclusion}"
        else:
            txt = fr" $\beta_{{\mathrm{{age}}}} = {beta:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} {p_perm_mapped}$" +  f"\n$BF_{{\\mathrm{{10}}}} = {BF10:.3f}, $" + f" {BF_conclusion}"

        ax.text(0.05, 1.2, txt, transform=ax.transAxes, fontsize=4, verticalalignment='top', linespacing=0.8)

        if BF_conclusion == 'strong H1' or BF_conclusion == 'moderate H1':
            sns.regplot(data=sub_df, x=sub_df['age_at_recording'] / 30, y=y_var,
                        marker='.', color="1", line_kws=dict(color="gray", lw=0.8), ax=ax)

        sns.scatterplot(data=sub_df, x=sub_df['age_at_recording'] / 30, y=y_var, hue='age_group',
                        palette=C.PALETTE, ax=ax, legend=False, marker='.')

        ax.set_ylabel(region)
        ax.set_xlabel("")
        ax.set_xticks([5, 10, 15, 20] if region in ['ACB', 'OLF', 'MBm', 'PO'] else [])
        ax.set_ylim(0, 180 if y_var == 'n_cluster' else 1.8)
        sns.despine(offset=2, trim=False, ax=ax)

    fig.suptitle(f"{y_var} across regions", fontsize=8)
    fig.supxlabel("Age (months)", fontsize=8).set_y(0.35)

    if save_fig:
        fname = f"F2S1_slice_org_{y_var}_{C.ALIGN_EVENT}_permutation_2025.pdf"
        fig.savefig(C.FIGPATH / fname)
        print(f"Saved figure to {C.FIGPATH}/{fname}")

    plt.show()


if __name__ == "__main__":
    print("Loading data...")
    neural_yield_table = load_neural_yield_table()

    for y_var in ['n_cluster', 'neural_yield']:
        permut_df = get_permut_results(y_var, C.AGE2USE, neural_yield_table)
        bf_df = get_bf_results(y_var, C.AGE2USE, neural_yield_table)
        plot_yield_by_region(neural_yield_table, permut_df, bf_df, y_var)


