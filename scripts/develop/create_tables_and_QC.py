#%% work with large tables and QC
from brainwidemap.bwm_loading import download_aggregate_tables, bwm_query, bwm_units, filter_sessions
from one.api import ONE
import pandas as pd
from learninglifespan_loading import lifespan_query
from preprocessing_functions import QC_filter, combine_regions
from pathlib import Path
import os
from iblatlas.regions import BrainRegions
from process_behavioral_data import filter_trials
from utils import config
one=ONE()
#%% download tables (bwm) or generate tables (lifespan)
datapath = '../data'
# STAGING_PATH = Path(one.cache_dir).joinpath('lifespan_tables')
datapath_bwm = Path(one.cache_dir).joinpath('bwm_tables')
datapath_lifespan = Path(one.cache_dir).joinpath('lifespan_tables')


bwm_df = bwm_query(alignment_resolved=True, return_details=False) #699 pids; 459 eids
lifespan_df = lifespan_query(one = one, alignment_resolved=True, return_details=False)#68 pids; 38 eids
bwm_df['project'] = 'brainwidemap'
lifespan_df['project'] = 'learninglifespan'
full_release_df = pd.concat([bwm_df,lifespan_df])

#load download_aggregate_tables for BWM
# clusters_table = download_aggregate_tables(one, type='clusters') 
# trials_table = download_aggregate_tables(one, type='trials')

clus_df_bwm = pd.read_parquet(os.path.join(datapath_bwm,'clusters.pqt'))
clus_df_lifespan = pd.read_parquet(os.path.join(datapath_lifespan,'clusters.pqt'))
# 按行合并（纵向合并），自动填充 NaN 处理非共有列

trials_df_bwm = pd.read_parquet(os.path.join(datapath_bwm,'trials.pqt'))
trials_df_lifespan = pd.read_parquet(os.path.join(datapath_lifespan,'trials.pqt'))
#%%
clus_df_merged = pd.concat([clus_df_bwm, clus_df_lifespan], axis=0, ignore_index=True)
trials_df_merged = pd.concat([trials_df_bwm, trials_df_lifespan], axis=0, ignore_index=True)
print(clus_df_merged.columns.nunique())
print(trials_df_merged.columns.nunique())
# print(clus_df_bwm.pid.nunique()) #699
# print(clus_df_bwm.eid.nunique()) #459
# print(trials_df.pid.nunique())
# print(trials_df_bwm.eid.nunique()) #459

#%% ========# actual QC process as shown in the table=======

#TODO: which dataset? bwm or lifespan or both?
merge_df=full_release_df
trials_df=trials_df_merged
clus_df=clus_df_merged
rt_range = (0.08, 0.2)
min_errors = 3
min_qc = 1.
min_sessions = 2
#%%
outcomes = []
######## level 1: session & insertion ##############
outcomes.append({
'name': 'Session and insertion QC',
'level': 'probe,session',
'n_sessions': merge_df.eid.nunique(), # query
'n_probes': merge_df.pid.nunique(), 
'n_units': clus_df.loc[clus_df['eid'].isin(merge_df.eid.unique())].uuids.nunique(),
})

######## level 2: trial： RT, event #############
# trials_df = trials_df.loc[trials_df['eid'].isin(mydf2.eid.unique())]
# mask trials_df using my function
#nan event & rt cutoff
trials_df["response_times_from_stim"] = trials_df["response_times"] - trials_df["stimOn_times"]
trials_df = filter_trials(trials_df, exclude_nan_event_trials=True, 
                                    trial_type='full', event_list=config.event_list, clean_rt=True, 
                                    rt_variable=config.rt_variable_name, rt_cutoff=config.rt_cutoff)

# trials_df = trials_df[trials_df['bwm_include']] # masked
merge_df = merge_df.loc[merge_df['eid'].isin(trials_df.eid.unique())] #index remaining sessions
outcomes.append({
    'name': 'Reaction time and missing events',
    'level': 'trials',
    'n_sessions': merge_df.eid.nunique(),
    'n_probes': merge_df.pid.nunique(),
    'n_units': clus_df.loc[clus_df['eid'].isin(merge_df.eid.unique())].uuids.nunique(),
}
)
######## level 3: session： 3 error trials #############
trials_agg = trials_df.groupby('eid').aggregate(
    n_trials=pd.NamedAgg(column='eid', aggfunc='count'), 
    n_error=pd.NamedAgg(column='feedbackType', aggfunc=lambda x: (x == -1).sum()),
)
trials_agg = trials_agg.loc[trials_agg['n_error'] >= min_errors]
eids_remain = trials_agg.index.to_list()
merge_df = merge_df.loc[merge_df['eid'].isin(eids_remain)]
outcomes.append({
    'name': 'Minimum 3 error trials',
    'level': 'session',
    'n_sessions': merge_df.eid.nunique(),
    'n_probes': merge_df.pid.nunique(),
    'n_units': clus_df.loc[clus_df['eid'].isin(merge_df.eid.unique())].uuids.nunique(),
})
    ######## level 4: unit： QC #############
#TODO: my good neuron!
clus_df = clus_df.loc[clus_df['eid'].isin(merge_df.eid.unique())]
clus_df = clus_df.loc[clus_df['label'] >= min_qc]
clus_df = clus_df.loc[clus_df['firing_rate'] > 1] # 2645(38 sessions)
clus_df = clus_df.loc[clus_df['presence_ratio'] > 0.95] # 1769(38 sessions)

outcomes.append({
    'name': 'Single unit QC',
    'level': 'neuron',
    'n_sessions': clus_df.eid.nunique(),
    'n_probes': clus_df.pid.nunique(),
    'n_units': clus_df.uuids.nunique(),
})
######## level 5: unit： ROI, #############
#limit to our ROIs from here
br = BrainRegions()
clus_df['Beryl'] = br.id2acronym(clus_df['atlas_id'], mapping='Beryl')
# clus_df = clus_df.loc[~clus_df[f'Beryl'].isin(['void', 'root'])]
clus_df['Beryl_merge'] = clus_df['Beryl'].copy()
clus_df['Beryl_merge'] = combine_regions(clus_df['Beryl_merge'].values)
# ROIs = ['PPC','visual_cortex','CA1','DG','PO','LP','midbrain_motor','SCm','MOs','ACA','limbic','ORB','olfactory_areas','ACB','CP','LS'] #,'PoT'
clus_df = clus_df.loc[clus_df['Beryl_merge'].isin(ROIs)]

outcomes.append({
    'name': 'ROIs only',
    'level': 'neuron',
    'n_sessions': clus_df.eid.nunique(),
    'n_probes': clus_df.pid.nunique(),
    'n_units': clus_df.uuids.nunique(),
})


######## level 7: region exclusion: at least 2 sessions #############
# Filter min 2 sessions per region 
units_count = clus_df.groupby([f'Beryl_merge', 'eid']).aggregate( # in each session, how many units exist in each region?
    n_units=pd.NamedAgg(column='cluster_id', aggfunc='count'),
)
#TODO: if i want to filter out the sessions with less than 5 units in each region, then i should use n_units
units_count = units_count.reset_index(level=['eid'])
region_df = units_count.groupby([f'Beryl_merge']).aggregate(
    n_sessions=pd.NamedAgg(column='eid', aggfunc='count'),
)
region_df = region_df[region_df['n_sessions'] >= min_sessions]
region_session_df = pd.merge(region_df, units_count, on=f'Beryl_merge', how='left')
region_session_df = region_session_df.reset_index(level=[f'Beryl_merge'])
region_session_df.drop(labels=['n_sessions', 'n_units'], axis=1, inplace=True)
clus_df = pd.merge(region_session_df, clus_df, on=['eid', f'Beryl_merge'], how='left')
outcomes.append({
    'name': 'Minimum 2 sessions per region',
    'level': 'neuron',
    'n_sessions': clus_df.eid.nunique(),
    'n_probes': clus_df.pid.nunique(),
    'n_units': clus_df.uuids.nunique(),
}) #TODO:check this criterion shouldn't filter any sessions if we only focus on our ROIs 
#############################
#  TODO: level 8: minimum 5 units per region? 
pids_remain = clus_df.pid.unique()
eids_remain = clus_df.eid.unique()

#%% save final results
release_df_remain = full_release_df.loc[full_release_df['pid'].isin(pids_remain)]
release_df_remain.to_csv(os.path.join(datapath,'BWM_LL_release_afterQC_df.csv'))

#%%
#%% load pids
try:
    # recordings_filtered = pd.read_parquet(os.path.join(datapath,'ibl_BWMLL_QCfiltered_probes_info.parquet'))
    recordings_filtered = pd.read_csv(os.path.join(datapath,'BWM_LL_release_afterQC_df.csv'))
except Exception as err:
    print(f'errored: {err}')