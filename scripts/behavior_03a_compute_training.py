"""
Compute training history.
TO examine the training process of both BWM and LearningLifespan mice

Input:  ibl_included_eids_trials_table2025_full.csv
Output: data/training_history_149subjs_2025_NEW_v2.parquet
"""
#%%
from one.api import ONE
import brainbox.behavior.training as training
from brainbox.io.one import SessionLoader
import pandas as pd
import numpy as np
import os
import config as C
from scripts.utils.io import read_table

one = ONE()
def merge_full_training_trials(data=None):

    training_table_list=[]
    for index, row in data.iterrows():
        subject = row['mouse_name']
        trials_table = one.load_aggregate('subjects', subject, '_ibl_subjectTrials.table')
        # Load training status and join to trials table
        training_table = one.load_aggregate('subjects', subject, '_ibl_subjectTraining.table')
        trials_table = (trials_table   
                .set_index('session')
                .join(training_table.set_index('session'))
                .sort_values(by=['session_start_time', 'intervals_0']))
        trials_table['training_status'] = trials_table.training_status.fillna(method='ffill')

        trials_table['subject']=subject
        trials_table = trials_table.reset_index()
        training_table_list.append(trials_table)
    all_subj_tt_df = pd.concat(training_table_list, ignore_index=True) 
    # if save_results:
    
    return all_subj_tt_df

def compute_training_history(data_info_unique, all_subj_tt_df):
    list_perf_easy = []
    not_included_subjs = []
    # failed_subjs = []

    # data_info_temp = data_info_unique.head()
    for index, row in data_info_unique.iterrows():
    # for index, row in data_info_unique_test.iterrows():
        print(f'processing {index + 1}/{len(data_info_unique)}') #
        subject = row['mouse_name']
        eid = row['eid']
        try:
            #access all available sessions(trials) from merged trialstable
            current_trials = all_subj_tt_df.loc[all_subj_tt_df['subject']==subject].reset_index().copy()
            current_trials['trials_date'] = current_trials['session_start_time'].dt.strftime('%Y-%m-%d') 

            eids_from_table = current_trials['session'].unique().tolist()
            if eid in eids_from_table:
                full_trials=current_trials
            else:
                eids_from_search = one.search(subject=subject) #, task_protocol=['biasedChoiceWorld','ephysChoiceWorld']
                assert len(eids_from_table) < len(eids_from_search), "there should be more sessions returned by one.search"
                # sess2add = list(filter(lambda x: x not in eids_from_table, eids_from_search))

                eids_from_search_str = set(str(eid) for eid in eids_from_search)
                sess2add = list(filter(lambda x: x not in eids_from_table, eids_from_search_str))

                df_sess2add = get_extra_trials(sess2add, current_trials['session_start_time'].min().date())
                df_sess2add['trials_date'] = df_sess2add['trials_date'].apply(lambda x: x.strftime('%Y-%m-%d'))

                if df_sess2add.shape[0]>0:
                    full_trials = pd.concat([current_trials, df_sess2add], axis=0, ignore_index=True)
                else:
                    full_trials=current_trials

            full_trials = (full_trials
                    .set_index('session')
                    .sort_values(by=['trials_date', 'intervals_0']))

            # full_trials['trials_date'] = full_trials['session_start_time'].dt.strftime('%Y-%m-%d')  #str, at date level
            full_trials['signed_contrast'] = training.get_signed_contrast(full_trials)
            full_trials['n_trials_session']=full_trials.groupby(['session'])['signed_contrast'].count()
            easy_trials = full_trials.loc[np.abs(full_trials['signed_contrast']) >= 50].reset_index()
            # grouped = easy_trials.groupby('trials_date')
            easy_trials['perf_easy'] = easy_trials.groupby('trials_date')['feedbackType'].transform(lambda x: np.sum(x == 1) / x.shape[0])

            #compress the df
            df_perf_easy = easy_trials[['session','trials_date','perf_easy','n_trials_session','training_status']].drop_duplicates()
            df_perf_easy['n_session'] = df_perf_easy.groupby(['trials_date'])['session'].transform('count')
            df_perf_easy['n_trials_day'] = df_perf_easy.groupby(['trials_date'])['n_trials_session'].transform('sum')
            #check the n_trials_day:
            print(df_perf_easy['n_trials_day'].min(), df_perf_easy['n_trials_day'].max())
            df_perf_easy = df_perf_easy[['trials_date','perf_easy','n_session','n_trials_day','training_status']].drop_duplicates().reset_index()
            # df_perf_easy = df_perf_easy
            #label the num_days_from_recording
            df_perf_easy['num_days_from_start']=np.linspace(1,df_perf_easy.shape[0],df_perf_easy.shape[0]) 
            # calculate num_days_from_recording/get trained
            df_perf_easy['num_days_from_recording']=np.nan
            df_perf_easy['num_days_from_trained']=np.nan
            try:
                idx_end_status = np.where(df_perf_easy['trials_date'] == row['date'])[0][0] #session!!!

                df_perf_easy.loc[idx_end_status, 'num_days_from_recording'] = 0
                df_perf_easy.loc[0:idx_end_status-1, 'num_days_from_recording'] = np.linspace(-idx_end_status, -1, idx_end_status)
                try:
                    # To use 'contains' method, replace NaN with ''
                    df_perf_easy['training_status'] = df_perf_easy['training_status'].fillna('')
                    date_trained_status = df_perf_easy[df_perf_easy['training_status'].str.contains("trained")]['trials_date'].iloc[0]
                    idx_trained_status = np.where(df_perf_easy['trials_date'] == date_trained_status)[0][0]
                    df_perf_easy.loc[idx_trained_status,'num_days_from_trained'] = 0
                    df_perf_easy.loc[0:idx_trained_status-1, 'num_days_from_trained'] = np.linspace(-idx_trained_status,-1,idx_trained_status) 
                except: print('skipping %s, could not identify first day trained'%subject); continue

                #Add subject info:
                df_perf_easy['mouse_name']=row['mouse_name']
                df_perf_easy['mouse_age']=row['mouse_age']
                list_perf_easy.append(df_perf_easy)

            except Exception as err1:
                print(subject, err1)
                not_included_subjs.append(subject)

        except Exception as err:
            print(f"Failed to load data for subject {subject}. Details: \n {err}")
            # failed_subjs.append(subject)

    training_table = pd.concat(list_perf_easy, ignore_index=True) 
    training_table['age_group'] = training_table['mouse_age'].apply(lambda age: 'old' if age > C.AGE_GROUP_THRESHOLD else 'young')

    return training_table


def get_extra_trials(sess2add, earliest_date):
    list_sess2add=[]
    for sess_eid in sess2add:
        try:
            sess_loader = SessionLoader(eid=sess_eid, one=one)
            sess_loader.load_trials()
            sess_trials = sess_loader.trials
            sess_n_trials = sess_trials.shape[0]
            sess_info = one.eid2ref(eid=sess_eid)
            sess_trials['session'] = sess_eid
            sess_trials['trials_date'] = one.eid2ref(eid=sess_eid).date
            # if (one.eid2ref(eid=sess_eid).date) > earliest_date: #TODO: Is it reasonable?
            list_sess2add.append(sess_trials)
        except Exception as err:
            print(sess_eid, err)
    df_sess = pd.concat(list_sess2add, ignore_index=True)
    return df_sess

def main(save_results=True):

    trials_table_file = C.DATAPATH / 'ibl_included_eids_trials_table2025_full.csv'
    trials_table = read_table(trials_table_file)#236118 trials

    #% filter sessions, for each mice, only keep the earlist recording session:
    session_info = trials_table[['eid','mouse_name','mouse_age','date']].drop_duplicates()
    # find the session_Date
    earliest_recording = session_info.groupby('mouse_name')['date'].min().reset_index()
    data_info_unique = pd.merge(earliest_recording, session_info, on=['mouse_name', 'date']) #146

    #% load merged trialstable across full training process 
    # merged_training_trials_file = 'subjectTrials_table_146subjs_updated.parquet'
    merged_training_trials_file = C.DATAPATH / 'subjectTraining_Trials_table_149subjs_2025.parquet'
    if os.path.exists(merged_training_trials_file):
        all_subj_tt_df = read_table(merged_training_trials_file)
    else:
        all_subj_tt_df = merge_full_training_trials(data=data_info_unique)
        all_subj_tt_df.to_parquet(merged_training_trials_file, engine='pyarrow', compression='snappy')

    training_table = compute_training_history(one, data_info_unique, all_subj_tt_df)
    
    if save_results:
        training_table['trials_date'] = pd.to_datetime(training_table['trials_date'])
        training_table.to_parquet(C.DATAPATH / 'training_history_149subjs_2025.parquet', engine='pyarrow', compression='snappy')


if __name__ == "__main__":
    main(save_results=True)
