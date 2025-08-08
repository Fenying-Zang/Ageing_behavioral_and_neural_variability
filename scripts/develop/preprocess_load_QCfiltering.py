"""
Fenying Zang, 2025

1. query all available probes (LearningLifespan+BWM) from database/released file 
2. merge all those probes
3. add extra QC steps 
4. save final filtered list of pids(and eids)
"""

#%% 
import pandas as pd
from pathlib import Path
import sys, os, time
from datetime import datetime
import tqdm
import numpy as np
import pickle
from one.api import ONE
from ibllib.atlas.regions import BrainRegions
from ibllib.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader
from preprocessing_functions import bwm_query, lifespan_query, filter_sessions, download_aggregate_tables, QC_filter

datapath = '../data'
datapath_bwm = Path(datapath).joinpath('bwm')
datapath_lifespan = Path(datapath).joinpath('lifespan')
one = ONE()

#['pid', 'eid', 'probe_name', 'session_number', 'date', 'subject', 'lab']
bwm_df = bwm_query(alignment_resolved=True, return_details=False, freeze='2023_12_bwm_release') #699 pids; 459 eids
bwm_df['project'] = 'brainwidemap'
lifespan_df = lifespan_query(one = one, alignment_resolved=True, return_details=False, freeze=None)#68 pids; 38 eids
lifespan_df['project'] = 'learninglifespan'
merge_df = pd.concat([bwm_df,lifespan_df])

#load download_aggregate_tables for BWM
# clusters_table_bwm = download_aggregate_tables(one, target_path=datapath_bwm, type='clusters')
# trials_table_bwm = download_aggregate_tables(one, target_path=datapath_bwm, type='trials')

trials_df_bwm = pd.read_parquet(os.path.join(datapath_bwm,'trials.pqt'))
clus_df_bwm = pd.read_parquet(os.path.join(datapath_bwm,'clusters.pqt'))
trials_df_lifespan = pd.read_parquet(os.path.join(datapath_lifespan,'trials.pqt'))
# trials_df_lifespan=trials_df_lifespan.loc[trials_df_lifespan['eid'].isin(lifespan_df['eid'])]
clus_df_lifespan = pd.read_parquet(os.path.join(datapath_lifespan,'clusters.pqt'))
#%% 
outcomes_bwm, filtered_pids_bwm, filtered_eids_bwm = QC_filter(
    merge_df=bwm_df, trials_df=trials_df_bwm, clus_df=clus_df_bwm,
    rt_range = (0.08, 0.2), min_errors = 3, min_qc = 1.,min_sessions = 2)

outcomes_lifespan, filtered_pids_lifespan, filtered_eids_lifespan = QC_filter(
    merge_df=lifespan_df, trials_df=trials_df_lifespan, clus_df=clus_df_lifespan,
    rt_range = (0.08, 0.2), min_errors = 3, min_qc = 1.,min_sessions = 2)

#%% merge 2 list of pids and eids
all_pids_filtered = filtered_pids_bwm.tolist() + filtered_pids_lifespan.tolist()
merge_df = merge_df.loc[merge_df['pid'].isin(all_pids_filtered)] 

#% exclude 2 sessions due to lack of age info
eids_noage = ['642c97ea-fe89-4ec9-8629-5e492ea4019d', 'caa5dddc-9290-4e27-9f5e-575ba3598614']
merge_df = merge_df.loc[~ merge_df['eid'].isin(eids_noage)] 
# print(merge_df['eid'].nunique(),merge_df['pid'].nunique())#355 eids, 476 pids

#% save the filtered pids (and basic info)
#['pid', 'eid', 'probe_name', 'session_number', 'date', 'subject', 'lab']
merge_df.to_parquet(os.path.join(datapath,'ibl_BWMLL_QCfiltered_probes_info.parquet'), engine='pyarrow', compression='snappy')

