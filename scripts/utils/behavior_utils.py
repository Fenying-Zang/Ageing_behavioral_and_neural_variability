"""
Utility functions for behavioral preprocessing and metrics.

Functions
---------
- create_trials_table : Extract trials for sessions, add metadata
- clean_rts           : Remove implausible RTs (too fast/slow)
- fit_psychfunc       : Fit a psychometric curve to choice data
- filter_trials       : Apply trial filters (events, RT cutoff, first 400)
- compute_choice_history : Add prev/next resp/feedback/contrast
- fit_psychometric_paras : Session-level psychometric + RT summary
"""
#%%
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import variation
from scipy import stats
from tqdm import tqdm


def create_trials_table(eids, one):
    """
    Extract all trials from the given sessions, add metadata, and return a DataFrame.

    Parameters
    ----------
    eids: Session eids to load from ONE
    one: ONE instance.

    Returns
    -------
    df_trials : pd.DataFrame
        Trial dataframe with subject/session metadata.
    err_list : list
        List of (eid, error) for sessions that failed to load.
    """
    all_trials = []
    err_list = []
    for i, eid in enumerate(tqdm(eids, desc="Processing eids")):
        print(f"[{i+1}/{len(eids)}] Loading {eid}")
        try:
            # Load trials object (prefer revision 2025-03-03 if available)
            try:
                trials_obj = one.load_object(eid, 'trials', revision='2025-03-03')
            except Exception:
                print(f"Revision '2025-03-03' not found for {eid}, loading default revision.")
                trials_obj = one.load_object(eid, 'trials')

            # Add signed contrast and aligned RTs
            trials_obj['signed_contrast'] = 100 * np.diff(np.nan_to_num(np.c_[trials_obj['contrastLeft'], 
                                                        trials_obj['contrastRight']]))
    
            trials_obj["response_times_from_stim"] = trials_obj["response_times"] - trials_obj["stimOn_times"]
            trials_obj["firstMovement_times_from_stim"] = trials_obj["firstMovement_times"] - trials_obj["stimOn_times"]
            
            # Convert to DataFrame
            trials = trials_obj.to_df() 
            trials['trial_index'] = trials.index # to keep track of choice history
            trials['response'] = trials['choice'].map({1: 0, 0: np.nan, -1: 1}) #-1: turning the wheel CCW; +1 turning CW
            trials['eid'] = eid

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
                age_at_recording, sex = np.nan, np.nan
            trials['mouse_age'] = age_at_recording
            trials['mouse_sex'] = sex
            trials['date'] = ref_dict.date

            all_trials.append(trials)
        except BaseException as e:
            print('Attention: ', eid, e)
            err_list.append((eid, e))

    df_trials = pd.concat(all_trials, ignore_index=True)   
    
    # keep only relevant columns
    df_trials = df_trials[['eid', 'mouse_name', 'mouse_age', 'mouse_sex', 'date', 'signed_contrast', 'probabilityLeft',
                           'goCue_times', 'stimOn_times', 'firstMovement_times', 'response', 'choice','response_times', 'feedback_times',
                           'response_times_from_stim', 'firstMovement_times_from_stim', 'feedbackType', 'trial_index']]

    return df_trials, err_list


def clean_rts(rt, cutoff=[0.08, 2]):
    """
    Clean reaction times by removing outliers (too fast/slow).

    Parameters
    ----------
    rt : pd.Series or np.ndarray
        Raw RT values.
    cutoff : [low, high]
        Acceptable range in seconds.

    Returns
    -------
    rt_clean : np.ndarray
        Cleaned RT values (invalid set to NaN).
    """
    assert (0 < np.nanmedian(rt) < 3) # median RT should be within some reasonable bounds

    print('cleaning RTs...')
    # remove RTs below and above cutoff
    rt_clean = rt.copy()
    rt_clean[rt_clean < cutoff[0]] = np.nan 
    rt_clean[rt_clean > cutoff[1]] = np.nan 

    return rt_clean


def fit_psychfunc(df):

    """
    Fit a psychometric function to choice data at given contrast levels.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'signed_contrast', 'choice', 'choice2'.

    Returns
    -------
    pd.DataFrame
        Fitted parameters: bias, threshold, lapses, ntrials.
    """    
    import psychofit as psy

    choicedat = df.groupby('signed_contrast').agg(
        {'choice': 'count', 'choice2': 'mean'}).reset_index()
    if len(choicedat) >= 4: # need some minimum number of unique x-values
        pars, L = psy.mle_fit_psycho(choicedat.values.transpose(), P_model='erf_psycho_2gammas',
                                 parstart=np.array(
                                     [0, 20., 0.05, 0.05]),
                                 parmin=np.array(
                                     [choicedat['signed_contrast'].min(), 5, 0., 0.]),
                                 parmax=np.array([choicedat['signed_contrast'].max(), 40., 1, 1]))
    else:
        pars = [np.nan, np.nan, np.nan, np.nan]

    df2 = {'bias': pars[0], 'threshold': pars[1],
           'lapselow': pars[2], 'lapsehigh': pars[3]}
    df2 = pd.DataFrame(df2, index=[0])

    df2['ntrials'] = df['choice'].count()

    return df2


def filter_trials(trials_df, exclude_nan_event_trials=True, trial_type ='first400', event_list=None, 
                  clean_rt=True, rt_variable=None,rt_cutoff=None):
    """
    Apply standard trial filters: drop missing events, subset trials, apply RT cutoff.

    Returns
    -------
    pd.DataFrame
        Filtered trial-level data with 'rt' and 'rt_raw' columns.
    """
    if exclude_nan_event_trials == True:
        trials_df['exclude_nan_event_mask'] = np.where(trials_df[event_list].notna().all(axis=1), 1, 0) # if 都非空，则1，否则0
        trials_df = trials_df.loc[trials_df['exclude_nan_event_mask']==1]
    if trial_type == 'first400':
        # groupby subject, pick only the first 400 trials 
        trials_df = trials_df.groupby('eid').head(400).reset_index(drop=True)
    elif trial_type == 'all':
        trials_df = trials_df.reset_index(drop=True)
    trials_df['rt_raw'] = trials_df[rt_variable].copy()
    trials_df['rt'] = clean_rts(trials_df[rt_variable], cutoff=rt_cutoff)    
    if clean_rt:# remove RTs that sit outside the cutoff window 
        trials_df = trials_df[~trials_df['rt'].isna()]
    return trials_df


def compute_choice_history(trials):
    """
    Add choice history columns (previous/next response, feedback, contrast).

    Parameters
    ----------
    trials : pd.DataFrame
        Must contain 'response', 'feedbackType', 'signed_contrast', 'trialnum'.

    Returns
    -------
    pd.DataFrame
        With extra columns for prev/next resp/fb/contrast.
    """
    print('adding choice history columns to database...')

    # append choice history 
    trials['prevresp'] = trials.response.shift(1)
    trials['prevfb'] = trials.feedbackType.shift(1)
    trials['prevcontrast'] = np.abs(trials.signed_contrast.shift(1))

    # also append choice future 
    trials['nextresp'] = trials.response.shift(-1)
    trials['nextfb'] = trials.feedbackType.shift(-1)
    trials['nextcontrast'] = np.abs(trials.signed_contrast.shift(-1))

    # remove when not consecutive based on trial_index
    trials_not_consecutive = (trials.trialnum - trials.trialnum.shift(1)) != 1.
    for col in ['prevresp', 'prevfb', 'prevcontrast']:
        trials.loc[trials_not_consecutive, col] = np.nan

    return trials


# def clean_rts(rt, cutoff=[0.08, 2],
#               compare_with=None, comparison_cutoff=None):

#     assert (0 < np.nanmedian(rt) < 3) # median RT should be within some reasonable bounds

#     print('cleaning RTs...')
#     # remove RTs below and above cutoff, for HDDM 
#     rt_clean = rt.copy()
#     rt_clean[rt_clean < cutoff[0]] = np.nan 
#     rt_clean[rt_clean > cutoff[1]] = np.nan 

#     # only keep RTs when they are close to the trial duration
#     if compare_with is not None:
#         timing_difference = compare_with - rt
#         # assert all(timing_difference > 0) # all RTs should be smaller than trial duration
#         rt_clean[timing_difference > comparison_cutoff] = np.nan

#     return rt_clean


# def rescale_contrast(x):
#     """
#     Since signed contrast does not linearly map onto drift rate, rescale it (with a basic tanh function)
#     to approximate linearity (so that we can include a single, linear 'contrast' term in the regression models)

#     See plot_contrast_rescale.py for a tanh fit, which generates the parameters below
#     """

#     a = 2.13731484
#     b = 0.05322221
    
#     return a * np.tanh( b * x )

def fit_psychometric_paras(data, split_type='block'):
    """
    Fit psychometric functions for each session and extract behavioral metrics.

    For each session, compute psychometric fits either by block (PLeft=0.2 vs 0.8) 
    or by previous response (prevresp=0 vs 1). Also compute combined fit.

    Adds session-level RT summary statistics (median, CV, MAD).

    Parameters
    ----------
    data : pd.DataFrame
        Trial-level data. Must include columns:
        ['eid', 'response', 'probabilityLeft', 'prevresp', 'feedbackType', 'rt', 'rt_raw'].
    split_type : {'block', 'prevresp'}
        How to split data for left vs right fits.

    Returns
    -------
    split_fits : pd.DataFrame
        Session-level metrics:
        - Psychometric parameters (threshold, bias, lapses) per condition and combined
        - Bias/lapse shifts between conditions
        - RT metrics: median, CV, MAD, raw/corrected
        - Aggregate measures: abs_bias, mean_lapse
    """
    from scipy.stats import variation

    data['choice2'] = data.response
    data['choice'] = data.response
    data['trial'] = data.index
    # cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100 
    split_fits = pd.DataFrame()
    for sess_eid in data['eid'].unique():
        if split_type == 'block':
            left_fit = fit_psychfunc(data[(data['eid'] == sess_eid)
                                        & (data['probabilityLeft'] == 0.8)])
            right_fit = fit_psychfunc(data[(data['eid'] == sess_eid)
                                        & (data['probabilityLeft'] == 0.2)]) 
            
        elif split_type == 'prevresp':
            left_fit = fit_psychfunc(data[(data['eid'] == sess_eid)
                                        & (data['prevresp'] == 0)]) #map({1: 0, 0: np.nan, -1: 1}) #-1: turning the wheel CCW; +1 turning CW l
            right_fit = fit_psychfunc(data[(data['eid'] == sess_eid)
                                        & (data['prevresp'] == 1)]) 
        
        combine_fit = fit_psychfunc(data[data['eid'] == sess_eid])
        
        #median rt
        corr_rt_median = (data['rt'][(data['eid'] == sess_eid) & (data['feedbackType'] == 1)]).median()
        rt_median = (data['rt'][data['eid'] == sess_eid]).median()
        rt_raw_median = (data['rt_raw'][data['eid'] == sess_eid]).median()
        
        #variability of rt:
        rt_CV = variation((data['rt'][data['eid'] == sess_eid]).values, ddof=1,nan_policy='omit')
        rt_raw_CV = variation((data['rt_raw'][data['eid'] == sess_eid]).values, ddof=1,nan_policy='omit')
        corr_rt_CV = variation((data['rt'][(data['eid'] == sess_eid) & (data['feedbackType'] == 1)]).values, ddof=1, nan_policy='omit')
        rt_MAD = stats.median_abs_deviation((data['rt'][data['eid'] == sess_eid]).values,nan_policy='omit')
        rt_raw_MAD = stats.median_abs_deviation((data['rt_raw'][data['eid'] == sess_eid]).values,nan_policy='omit')
        corr_rt_MAD = stats.median_abs_deviation((data['rt'][(data['eid'] == sess_eid) & (data['feedbackType'] == 1)]).values,nan_policy='omit')
        
        # Combine results
        data_temp={'eid': sess_eid,
                    'threshold_l': left_fit['threshold'],
                    'threshold_r': right_fit['threshold'],
                    'threshold': combine_fit['threshold'],
                    'bias_l': left_fit['bias'],
                    'bias_r': right_fit['bias'],
                    'bias': combine_fit['bias'],
                    'lapselow_l': left_fit['lapselow'],
                    'lapselow_r': right_fit['lapselow'],
                    'lapselow': combine_fit['lapselow'],
                    'lapsehigh_l': left_fit['lapsehigh'],
                    'lapsehigh_r': right_fit['lapsehigh'],
                    'lapsehigh': combine_fit['lapsehigh'],
                    'corr_rt_median':corr_rt_median,
                    'rt_median':rt_median,
                    'rt_raw_median':rt_raw_median,
                    'rt_CV':rt_CV,
                    'rt_raw_CV':rt_raw_CV,
                    'corr_rt_CV':corr_rt_CV,
                    'rt_MAD':rt_MAD,
                    'rt_raw_MAD':rt_raw_MAD,
                    'corr_rt_MAD':corr_rt_MAD}
        fits = pd.DataFrame(data_temp)
        split_fits = split_fits._append(fits, sort=False)  
    
    # Derived shift and summary metrics
    split_fits['bias_shift'] = split_fits['bias_l'] - split_fits['bias_r']
    split_fits['lapselow_shift'] = split_fits['lapselow_l'] - split_fits['lapselow_r']
    split_fits['lapsehigh_shift'] = split_fits['lapsehigh_l'] - split_fits['lapsehigh_r']

    split_fits['abs_bias'] = split_fits['bias'].abs()
    split_fits['mean_lapse'] = (split_fits['lapsehigh']+split_fits['lapselow'])/2
    return split_fits
