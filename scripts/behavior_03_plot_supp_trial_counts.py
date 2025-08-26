"""
Figure: Fig1-S2 — Trial counts

input:	data/ibl_included_eids_trials_table2025_full.csv
output: figures/F1S2_num_trials_filtering.pdf

"""
#%%
from ibl_style.utils import get_coords, MM_TO_INCH, double_column_fig
from scripts.utils.plot_utils import figure_style
import figrid as fg
from one.api import ONE
import pandas as pd
import seaborn as sns
from scripts.utils.behavior_utils import filter_trials
from scripts.utils.plot_utils import format_bf_annotation
from scripts.utils.data_utils import add_age_group

import config as C
from scripts.utils.stats_utils import get_bf_results, get_permut_results
from scripts.utils.io import read_table, save_figure
import logging


log = logging.getLogger(__name__)
one = ONE()


# Define the default styling used for figures
def setup_fig_axes(fg, MM_TO_INCH, fig=None):
    if fig is None:
        fig = double_column_fig()
    figure_style()

    # Make a double column figure
    fig = double_column_fig()

    # Get the dimensions of the figure in mm
    width, height = fig.get_size_inches() / MM_TO_INCH  #180, 170
    yspans = get_coords(height, ratios=[1,1], space=15, pad=5, span=(0, 0.6))

    xspans1 = get_coords(width, ratios=[1, 1, 1], space=20, pad=5, span=(0, 1))  #from 0-1

    axs = {'scatter_raw_trials': fg.place_axes_on_grid(fig, xspan=xspans1[0], yspan=yspans[0]),
            'scatter_clean_trials': fg.place_axes_on_grid(fig, xspan=xspans1[2], yspan=yspans[0]),
            'scatter_num_exc_trials': fg.place_axes_on_grid(fig, xspan=xspans1[1], yspan=yspans[0]),

    }
    return fig, axs


def plot_single_scatterplot(df, ax, p_perm, observed_val, BF10, BF_conclusion=None):   
    
    txt = format_bf_annotation(observed_val, p_perm, BF10, BF_conclusion, beta_label="age", big_bf=100)
    ax.text(0.05, 1.1, txt, transform=ax.transAxes, fontsize=6, verticalalignment='top')

    if BF_conclusion == 'strong H1' or BF_conclusion == 'moderate H1':
        sns.regplot(data=df, x='age_months', y='n_trials', 
                    marker='.', color="1", line_kws=dict(color="gray"), ax=ax)
    else:
        sns.regplot(data=df, x='age_months', y='n_trials', 
                    fit_reg=False, marker='.', color="1", line_kws=dict(color="gray"), ax=ax) #color = .3
    sns.scatterplot(x='age_months', y='n_trials', data=df, hue='age_group',
            alpha=1, marker='.', legend=False, palette=C.PALETTE, hue_order=['young','old'], ax=ax)  #marker='o',s=10,

    ax.set_xlabel('Age (months)')
    ax.set_ylabel('# trials')  
    sns.despine(offset=2, trim=False, ax=ax)
    return ax


def main(save_fig=True):
    trials_table_file = C.DATAPATH / 'ibl_included_eids_trials_table2025_full.csv'
    trials_table = read_table(trials_table_file)

    trials_table_filltered = filter_trials(
        trials_table, exclude_nan_event_trials=True, 
        trial_type=C.TRIAL_TYPE, event_list=C.EVENT_LIST, 
        clean_rt=False, rt_variable=C.RT_VARIABLE_NAME,  #do not exclude nan here
        rt_cutoff=C.RT_CUTOFF)
    
    fig, axs = setup_fig_axes(fg, MM_TO_INCH)
    
    for content in ['raw', 'clean']:
        ax = axs[f'scatter_{content}_trials']
        if content == 'raw':
            data2fit = trials_table
        else: 
            data2fit = trials_table_filltered.loc[~trials_table_filltered['rt'].isna()] 
        data2fit = add_age_group(data2fit)
        num_trials_df = data2fit.groupby(['age_group', 'eid', 'mouse_age']).aggregate(
                                    n_trials=pd.NamedAgg(column='trial_index', aggfunc='count'),
                                    ).reset_index()
        num_trials_df = add_age_group(num_trials_df)
        age2use='age_years'
        permut_filename = C.RESULTSPATH / f"t_num_{content}_trials_{age2use}_{C.N_PERMUT_BEHAVIOR}permutation.csv"
        p_perm, observed_val = get_permut_results(content, age2use='age_years', df=num_trials_df, filename=permut_filename) #TODO: 
        BF_filename = C.RESULTSPATH / f"t_beyesfactor_{content}_trials.csv"
        BF10, BF_conclusion = get_bf_results(content, df=num_trials_df, age2use='age_years', filename=BF_filename)
        ax = plot_single_scatterplot(num_trials_df, ax, p_perm, observed_val, BF10, BF_conclusion)
    
    ax = axs['scatter_num_exc_trials']
    data2fit = trials_table_filltered.loc[trials_table_filltered['rt'].isna()] 
    data2fit = add_age_group(data2fit)

    num_trials_df_exc = data2fit.groupby(['age_group','eid','mouse_age']).aggregate(
                                n_trials=pd.NamedAgg(column='trial_index', aggfunc='count'),
                                ).reset_index()
    num_trials_df_exc['age_months'] = num_trials_df_exc['mouse_age']/30
    num_trials_df_exc['age_years'] = num_trials_df_exc['mouse_age']/365

    p_perm, observed_val = get_permut_results('exc', age2use='age_years', df=num_trials_df_exc) #TODO:
    BF10, BF_conclusion = get_bf_results('exc', df=num_trials_df_exc, age2use='age_years')

    ax = plot_single_scatterplot(num_trials_df_exc, ax, p_perm, observed_val, BF10, BF_conclusion)
    if save_fig:
        figname = C.FIGPATH / "F1S2_num_trials_filtering.pdf"
        save_figure(fig, figname, add_timestamp=True)


if __name__ == "__main__":
    from scripts.utils.io import setup_logging
    setup_logging()

    main(save_fig=True)
