"""

input:	data/ibl_included_eids_trials_table2025_full.csv
output: figures/F1S1_num_trials_filtering.pdf

"""
#%%
from ibl_style.utils import get_coords, MM_TO_INCH, double_column_fig
from ibl_style.style import figure_style
import figrid as fg
import matplotlib.pyplot as plt
from one.api import ONE
import pandas as pd
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from statsmodels.genmod.families import Gaussian
from statsmodels.formula.api import glm
from tqdm import tqdm
from scripts.utils.behavior_utils import filter_trials
from scripts.utils.plot_utils import plot_permut_test, map_p_value
from scripts.utils.data_utils import shuffle_labels_perm, bf_gaussian_via_pearson, interpret_bayes_factor
import config as C

one = ONE()
# Define the default styling used for figures
def setup_fig_axes(fg, MM_TO_INCH, fig=None):
    if fig is None:
        fig = double_column_fig()
    figure_style()

       # Make a double column figure
    fig = double_column_fig()

       # Get the dimensions of the figure in mm
    width, height = fig.get_size_inches() / MM_TO_INCH #180, 170
    yspans = get_coords(height, ratios=[1,1], space=15, pad=5, span=(0, 0.6))

    xspans1 = get_coords(width, ratios=[1, 1, 1], space=20, pad=5, span=(0, 1))#from 0-1

    axs = {'scatter_raw_trials': fg.place_axes_on_grid(fig, xspan=xspans1[0], yspan=yspans[0]),
              'scatter_clean_trials': fg.place_axes_on_grid(fig, xspan=xspans1[2], yspan=yspans[0]),
              'scatter_num_exc_trials': fg.place_axes_on_grid(fig, xspan=xspans1[1], yspan=yspans[0]),

    }
    return fig, axs

# def load_trial_table(filepath):
#     trials_table = pd.read_csv(filepath)
#     print(len(set(trials_table['eid'])), 'sessions loaded')
#     return trials_table


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

def get_bf_results(content, df, age2use):
    filename = C.DATAPATH / f"beyesfactor_{content}_trials.csv"
    if filename.exists():
        BF_dict = pd.read_csv(filename)
        BF10 = BF_dict['BF10'].values[0]
        BF_conclusion = BF_dict['BF_conclusion'].values[0]
    else:

        BF_dict = bf_gaussian_via_pearson(df, 'n_trials', age2use)
        print(f"BF10 for {content} vs. {age2use}: {BF_dict['BF10']:.3f}")
        BF10 = BF_dict['BF10']
        BF_conclusion = interpret_bayes_factor(BF10)
        bf_df = pd.DataFrame({
            'y_col': content,
            'BF10': BF10,
            'BF_conclusion': BF_conclusion
        }, index=[0])
        bf_df.to_csv(filename, index=False)
    return BF10, BF_conclusion

def get_permut_results (content, age2use, df):

    filename = C.DATAPATH / f"num_{content}_trials_{age2use}_{C.N_PERMUT_BEHAVIOR}permutation.csv"
    if filename.exists():
        permut_df = pd.read_csv(filename)
        p_perm = permut_df['p_perm'].values[0]
        observed_val = permut_df['observed_val'].values[0]
    else:
        formula2use = f"n_trials ~ {age2use} "
        this_age = df[age2use].values
        family_func = Gaussian()
        observed_val, observed_val_p, p_perm, valid_null = run_permutation_test(
            data=df,
            this_age=this_age,
            formula2use=formula2use,
            family_func=family_func,
            n_permut=C.N_PERMUT_BEHAVIOR,
            n_jobs=6
        )
        print(f"Omnibus results for num_{content}_trials: \n  beta = {observed_val:.4f}, p_perm = {p_perm:.4f}")
        # if plot_permt_result:
        #     plot_permut_test(null_dist=valid_null, observed_val=observed_val, p=p_perm, mark_p=None, metric=y_var, save_path=C.FIGPATH, show=True, region=region)

        permut_df = pd.DataFrame({
            'y_col': content,
            'n_perm': C.N_PERMUT_BEHAVIOR,
            'formula': formula2use,
            'observed_val': observed_val,
            'observed_val_p': observed_val_p,
            'p_perm': p_perm,
            'ave_null_dist': valid_null.mean()
        }, index=[0]) 
        permut_df.to_csv(filename, index=False)
    return p_perm, observed_val


def plot_single_scatterplot(df, ax, p_perm, observed_val, BF10, BF_conclusion=None):   
    
    p_perm_mapped = map_p_value(p_perm)

    if BF10 > 100:
        txt = fr" $\beta_{{\mathrm{{age}}}} = {observed_val:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} {p_perm_mapped}$" +  f"\n$BF_{{\\mathrm{{10}}}} > 100, $" + f" {BF_conclusion}"
    else:
        txt = fr" $\beta_{{\mathrm{{age}}}} = {observed_val:.3f}, $"+ f"$p_{{\\mathrm{{perm}}}} {p_perm_mapped}$" +  f"\n$BF_{{\\mathrm{{10}}}} = {BF10:.3f}, $" + f" {BF_conclusion}"

    ax.text(0.05, 1.1, txt , transform=ax.transAxes, fontsize=6,verticalalignment='top')

    if BF_conclusion == 'strong H1' or BF_conclusion == 'moderate H1':
        sns.regplot(data=df, x='age_months', y='n_trials', 
                    marker='.', color="1", line_kws=dict(color="gray"), ax=ax)
    else:
        sns.regplot(data=df, x='age_months', y='n_trials', 
                    fit_reg=False, marker='.', color="1", line_kws=dict(color="gray"), ax=ax) #color = .3
    sns.scatterplot(x='age_months', y='n_trials', data=df, hue='age_group',
            alpha=1, marker='.',legend=False, palette=C.PALETTE, hue_order=['young','old'], ax=ax)  #marker='o',s=10,

    ax.set_xlabel('Age (months)')
    ax.set_ylabel('# trials')  
    sns.despine(offset=2, trim=False, ax=ax)
    return ax


def main(save_fig = True):

    trials_table_file = C.DATAPATH / 'ibl_included_eids_trials_table2025_full.csv'
    trials_table = load_trial_table(trials_table_file)

    trials_table_filltered = filter_trials(
        trials_table, exclude_nan_event_trials=True, 
        trial_type=C.TRIAL_TYPE, event_list=C.EVENT_LIST, 
        clean_rt=False, rt_variable=C.RT_VARIABLE_NAME,  #do not exclude nan here
        rt_cutoff=C.RT_CUTOFF)
    
    # permut_results = load_permut_results() #TODO
    fig, axs = setup_fig_axes(fg, MM_TO_INCH)
    
    for content in ['raw','clean']:
        ax = axs[f'scatter_{content}_trials']
        if content=='raw':
            data2fit = trials_table
        else: 
            data2fit = trials_table_filltered.loc[~trials_table_filltered['rt'].isna()] 
        data2fit['age_group'] = (data2fit['mouse_age'] > C.AGE_GROUP_THRESHOLD).map({True: "old", False: "young"})
        
        num_trials_df = data2fit.groupby(['age_group','eid','mouse_age']).aggregate(
                                    n_trials = pd.NamedAgg(column='trial_index',aggfunc='count'),
                                    ).reset_index()
        num_trials_df['age_months'] = num_trials_df['mouse_age']/30
        num_trials_df['age_years'] = num_trials_df['mouse_age']/365
        p_perm, observed_val = get_permut_results (content, age2use='age_years', df=num_trials_df) #TODO: 
        BF10, BF_conclusion = get_bf_results(content, df=num_trials_df,age2use='age_years')
        ax = plot_single_scatterplot(num_trials_df, ax, p_perm, observed_val, BF10, BF_conclusion)
    
    ax = axs['scatter_num_exc_trials']

    data2fit = trials_table_filltered.loc[trials_table_filltered['rt'].isna()] 
    data2fit['age_group'] = (data2fit['mouse_age'] > C.AGE_GROUP_THRESHOLD).map({True: "old", False: "young"})

    num_trials_df_exc = data2fit.groupby(['age_group','eid','mouse_age']).aggregate(
                                n_trials = pd.NamedAgg(column='trial_index',aggfunc='count'),
                                ).reset_index()
    num_trials_df_exc['age_months'] = num_trials_df_exc['mouse_age']/30
    num_trials_df_exc['age_years'] = num_trials_df_exc['mouse_age']/365

    p_perm, observed_val = get_permut_results ('exc', age2use='age_years', df=num_trials_df_exc) #TODO:
    BF10, BF_conclusion = get_bf_results('exc', df=num_trials_df_exc,age2use='age_years')

    ax = plot_single_scatterplot(num_trials_df_exc, ax, p_perm, observed_val, BF10, BF_conclusion)
    if save_fig:
        figname = C.FIGPATH / f"F1S1_num_trials_filtering.pdf"
        fig.savefig(figname, dpi=300)
        print(f"Figure saved to {C.FIGPATH}")


if __name__ == "__main__":
    main(save_fig = True)

#%%