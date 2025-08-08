"""
Fenying Zang, 2024

1. load the existing trials table (or generate trials table) for all sessions included
    a. trials info, subject info
    b. calculate RT
    c. concat all
2. filtering trials (nan_events, first 400, rt outlier)

"""
#%% 
import pandas as pd
import numpy as np
from datetime import datetime
from utils.behavior_utils import fit_psychfunc, fit_psychometric_paras, clean_rts
from one.api import ONE
import os, sys
one= ONE()

#%%
def create_trials_table(eids, one):
    """ 
    extract all trials of those sessions,
    add info of subject, return a dataframe

    Args:
        eids (_type_): _description_

    Returns:
        df_trials: 
        err
    """
    all_trials = []
    err_list = []
    for _, eid in enumerate(eids):
        try:
            try:
                trials = one.load_object(eid, 'trials').to_df()
            except Exception as e:
                print(e)
                trials = one.load_object(eid, 'trials',revision='2024-07-15').to_df()
            trials['signed_contrast'] = 100 * np.diff(np.nan_to_num(np.c_[trials['contrastLeft'], 
                                                        trials['contrastRight']]))
            trials['contrast'] = (trials['signed_contrast'].abs())/100
            trials['side'] = np.where(trials['contrastRight'].notna(), 'right', 'left')

            # TODO: go_Cue or stimulus onset? 
            # trials_obj['trial_duration'] = trials_obj['response_times'] - trials_obj['goCue_times']
            trials["response_times_from_stim"] = trials["response_times"] - trials["stimOn_times"]
            trials["firstMovement_times_from_stim"] = trials["firstMovement_times"] - trials["stimOn_times"]
   
            # trials = trials_obj.to_df() # to dataframe
            trials['trial_index'] = trials.index # to keep track of choice history
            trials['response'] = trials['choice'].map({1: 0, 0: np.nan, -1: 1}) #-1: turning the wheel CCW; +1 turning CW
            trials['eid'] = eid
            # trials.insert(0, 'eid', eid)
            # retrieve the mouse name, session etc
            ref_dict = one.eid2ref(eid)
            
            trials['mouse_name'] = ref_dict.subject
            # sess_date = ref_dict.date
            session_details = one.get_details(eid)
            sess_date = session_details['start_time'][:10]

            try:
                subj = one.alyx.rest('subjects', 'list', nickname=ref_dict.subject)
                subj_dob = subj[0]['birth_date']
                sex = subj[0]['sex']
                age_at_recording = (datetime.strptime(sess_date, '%Y-%m-%d') - datetime.strptime(subj_dob, '%Y-%m-%d')).days
            except Exception:
                age_at_recording = np.nan
            trials['mouse_age'] = age_at_recording
            trials['mouse_sex'] = sex
            trials['date'] = ref_dict.date 

            all_trials.append(trials)
        except BaseException as e:
            print(eid, e)
            err_list.append((eid, e))

    df_trials = pd.concat(all_trials, ignore_index=True)   
    
    #only with some columns we need 
    df_trials = df_trials[['eid', 'mouse_name', 'mouse_age','mouse_sex','date', 'signed_contrast','contrast','side','probabilityLeft',
                           'goCue_times','stimOn_times','firstMovement_times','response', 'choice','response_times','feedback_times',
                           'response_times_from_stim','firstMovement_times_from_stim','feedbackType', 'trial_index']]

    return df_trials, err_list


def filter_trials(trials_df, exclude_nan_event_trials=True, trial_type='first400', event_list=None, 
                  clean_rt=True, rt_variable=None,rt_cutoff=None):
    # mask nan values; pick first 400 trials; rt cutoff 
    if exclude_nan_event_trials == True:
        trials_df['exclude_nan_event_mask'] = np.where(trials_df[event_list].notna().all(axis=1), 1, 0) # if 都非空，则1，否则0
        trials_df = trials_df.loc[trials_df['exclude_nan_event_mask']==1]
    if trial_type == 'first400':
        # groupby subject, pick only the first 400 trials 
        trials_df = trials_df.groupby('eid').head(400).reset_index(drop=True)
    trials_df['rt_raw'] = trials_df[rt_variable].copy()
    trials_df['rt'] = clean_rts(trials_df[rt_variable], cutoff=rt_cutoff)    
    if clean_rt == True:# remove RTs that sit outside the cutoff window 
        trials_df = trials_df[~trials_df['rt'].isna()]
    return trials_df

# # %% #create a merged trials table for all sessions
# datapath='../data'
# try:
#     # recordings_filtered = pd.read_parquet(os.path.join(datapath,'ibl_BWMLL_QCfiltered_probes_info.parquet'))
#     recordings_filtered = pd.read_csv(os.path.join(datapath,'BWM_LL_release_afterQC_df.csv'))
# except Exception as err:
#     print(f'errored: {err}')
# eids = recordings_filtered['eid'].astype(str).unique().tolist()

# trials_table, err_list = create_trials_table(eids, one)
# trials_table['age_group'] = (trials_table['mouse_age'] > 300).map({True: "old", False: "young"})
# #%%
# # trials_table_file = os.path.join(datapath, 'ibl_included_eids_trials_table.csv') 
# trials_table_file = os.path.join(datapath, 'ibl_included_eids_trials_table2025_full.csv') 
# trials_table.to_csv(trials_table_file)
# print('new trial table saved here: ', trials_table_file)

# #%%
# # print('%d mice, %d trials'%(trials_table.mouse_name.nunique(),  trials_table.mouse_name.count()))  #147 mice, 236118 trials
# #%%
# # trials_table = trials_table.drop_duplicates(keep='first')
# # print(len(trials_table.eid))
# # print(trials_table.eid.nunique())

# print(err_list)
# # [('69c9a415-f7fa-4208-887b-1417c1479b48', ValueError('Can only convert to DataFrame objects with consistent size'))]
# %%
