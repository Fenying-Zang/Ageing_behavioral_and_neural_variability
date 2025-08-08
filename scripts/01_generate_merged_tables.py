
"""
generate tables: trials table; training table

input: 
output: ibl_included_eids_trials_table2025_full.csv

"""
#%%
import os
import pandas as pd

import numpy as np
from datetime import datetime
# from utils.behavior_utils import fit_psychfunc, fit_psychometric_paras, clean_rts
from one.api import ONE
import os, sys
from utils.data_utils import load_filtered_recordings
from utils.behavior_utils import create_trials_table
from utils import config

if __name__ == "__main__":
    
    one= ONE()
    datapath = config.datapath

    #load filtered eid list
    recordings_filtered = load_filtered_recordings(datapath)
    eids = recordings_filtered['eid'].astype(str).unique().tolist()
    # eids = ['85dc2ebd-8aaf-46b0-9284-a197aee8b16f','f88d4dd4-ccd7-400e-9035-fa00be3bcfa8']
    print(len(eids))

    #merged trials table:
    trials_table, err_list = create_trials_table(eids[0:40], one)
    trials_table['age_group'] = (trials_table['mouse_age'] > config.age_group_threshold).map({True: "old", False: "young"})
    trials_table_file = os.path.join(datapath, 'ibl_included_eids_trials_table2025_full.csv') 
    trials_table.to_csv(trials_table_file)
    
    print('new trial table saved here: ', trials_table_file)
    print(f'{trials_table.mouse_name.nunique()} mice, {trials_table.mouse_name.count()} trials')  #147 mice, 236118 trials
#%%
# trials_table = trials_table.drop_duplicates(keep='first')
# print(len(trials_table.eid))
# print(trials_table.eid.nunique())
