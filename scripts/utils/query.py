"""
Query helpers for BWM and LearningLifespan datasets.

Functions
---------
- bwm_query      : Query brainwide map insertions passing core QC or load a frozen list.
- lifespan_query : Query learninglifespan insertions passing core QC.

Notes
-----
- Both functions return a DataFrame with one row per insertion:
  ['pid', 'eid', 'probe_name', 'session_number', 'date', 'subject', 'lab', ...]
- The QC filters are implemented via alyx django query strings.
- `marked_pass` block includes insertions manually marked PASS by experimenters,
  in addition to those passing automated extended QC thresholds.
"""

from dateutil import parser
import numpy as np
import pandas as pd
from iblutil.numerical import ismember
import config as C


def bwm_query(one=None, alignment_resolved=True, return_details=False, freeze='2023_12_bwm_release'):
    """
    Function to query for brainwide map sessions that pass the most important quality controls. Returns a dataframe
    with one row per insertions and columns ['pid', 'eid', 'probe_name', 'session_number', 'date', 'subject', 'lab']

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to local or remote database. Only required if freeze=None.
    alignment_resolved: bool
        Default is True. If True, only returns sessions with resolved alignment, if False returns all sessions with at
        least one alignment
    return_details: bool
        Default is False. If True returns a second output a list containing the full insertion dictionary for all
        insertions returned by the query. Only needed if you need information that is not contained in the bwm_df.
    freeze: {None, 2022_10_initial, 2022_10_update, 2022_10_bwm_release, 2023_12_bwm_release}
        Default is 2023_12_bwm_release. If None, the database is queried for the current set of pids satisfying the
        criteria. If a string is specified, a fixed set of eids and pids is returned instead of querying the database.

    Returns
    -------
    bwm_df: pandas.DataFrame
        BWM sessions to be included in analyses with columns
        ['pid', 'eid', 'probe_name', 'session_number', 'date', 'subject', 'lab']
    insertions: list
        Only returned if return_details=True. List of dictionaries with details for each insertions.
    """

    # If a freeze is requested just try to load the respective file
    if freeze is not None:
        if return_details is True:
            print('Cannot return details when using a data freeze. Returning only main dataframe.')
            
        freeze_file = C.DATAPATH / f'{freeze}.csv'
        print(f'Loading bwm_query results from {freeze}.csv')
        bwm_df = pd.read_csv(freeze_file, index_col=0)
        bwm_df['date'] = [parser.parse(i).date() for i in bwm_df['date']]

        return bwm_df

    # Else, query the database
    assert one is not None, 'If freeze=None, you need to pass an instance of one.api.ONE'
    base_query = (
        'session__projects__name__icontains,ibl_neuropixel_brainwide_01,'
        '~session__json__IS_MOCK,True,'
        'session__qc__lt,50,'
        'session__extended_qc__behavior,1,'
        '~json__qc,CRITICAL,'
        'json__extended_qc__tracing_exists,True,'
    )

    if alignment_resolved:
        base_query += 'json__extended_qc__alignment_resolved,True,'
    else:
        base_query += 'json__extended_qc__alignment_count__gt,0,'

    qc_pass = (
        '~session__extended_qc___task_stimOn_goCue_delays__lt,0.9,'
        '~session__extended_qc___task_response_feedback_delays__lt,0.9,'
        '~session__extended_qc___task_wheel_move_before_feedback__lt,0.9,'
        '~session__extended_qc___task_wheel_freeze_during_quiescence__lt,0.9,'
        '~session__extended_qc___task_error_trial_event_sequence__lt,0.9,'
        '~session__extended_qc___task_correct_trial_event_sequence__lt,0.9,'
        '~session__extended_qc___task_reward_volumes__lt,0.9,'
        '~session__extended_qc___task_reward_volume_set__lt,0.9,'
        '~session__extended_qc___task_stimulus_move_before_goCue__lt,0.9,'
        '~session__extended_qc___task_audio_pre_trial__lt,0.9')

    marked_pass = (
        'session__extended_qc___experimenter_task,PASS')

    insertions = list(one.alyx.rest('insertions', 'list', django=base_query + qc_pass))
    insertions.extend(list(one.alyx.rest('insertions', 'list', django=base_query + marked_pass)))

    bwm_df = pd.DataFrame({
        'pid': np.array([i['id'] for i in insertions]),
        'eid': np.array([i['session'] for i in insertions]),
        'probe_name': np.array([i['name'] for i in insertions]),
        'session_number': np.array([i['session_info']['number'] for i in insertions]),
        'date': np.array([parser.parse(i['session_info']['start_time']).date() for i in insertions]),
        'subject': np.array([i['session_info']['subject'] for i in insertions]),
        'lab': np.array([i['session_info']['lab'] for i in insertions]),
        'task_protocol':np.array([i['session_info']['task_protocol'] for i in insertions]),
    }).sort_values(by=['lab', 'subject', 'date', 'eid'])
    bwm_df.drop_duplicates(inplace=True)
    bwm_df.reset_index(inplace=True, drop=True)

    if return_details:
        return bwm_df, insertions
    else:
        return bwm_df


def lifespan_query(one=None, alignment_resolved=True, return_details=False):
    """
    Function to query for learninglifespan sessions that pass the most important quality controls. Returns a dataframe
    with one row per insertions and columns ['pid', 'eid', 'probe_name', 'session_number', 'date', 'subject', 'lab']

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to local or remote database. Only required if freeze=None.
    alignment_resolved: bool
        Default is True. If True, only returns sessions with resolved alignment, if False returns all sessions with at
        least one alignment
    return_details: bool
        Default is False. If True returns a second output a list containing the full insertion dictionary for all
        insertions returned by the query. Only needed if you need information that is not contained in the bwm_df.

    Returns
    -------
    ll_df: pandas.DataFrame
        LL sessions to be included in analyses with columns
        ['pid', 'eid', 'probe_name', 'session_number', 'date', 'subject', 'lab']
    insertions: list
        Only returned if return_details=True. List of dictionaries with details for each insertions.
    """

    # Else, query the database
    assert one is not None, 'If freeze=None, you need to pass an instance of one.api.ONE'
    base_query = (
        'session__projects__name,churchland_learninglifespan,'
        '~session__json__IS_MOCK,True,' 
        'session__qc__lt,50,'
        'session__extended_qc__behavior,1,'
        '~json__qc,CRITICAL,'
        'json__extended_qc__tracing_exists,True,'
    )

    if alignment_resolved:
        base_query += 'json__extended_qc__alignment_resolved,True,'
    else:
        base_query += 'json__extended_qc__alignment_count__gt,0,'

    qc_pass = (
        '~session__extended_qc___task_stimOn_goCue_delays__lt,0.9,'
        '~session__extended_qc___task_response_feedback_delays__lt,0.9,'
        '~session__extended_qc___task_wheel_move_before_feedback__lt,0.9,'
        '~session__extended_qc___task_wheel_freeze_during_quiescence__lt,0.9,'
        '~session__extended_qc___task_error_trial_event_sequence__lt,0.9,'
        '~session__extended_qc___task_correct_trial_event_sequence__lt,0.9,'
        '~session__extended_qc___task_reward_volumes__lt,0.9,'
        '~session__extended_qc___task_reward_volume_set__lt,0.9,'
        '~session__extended_qc___task_stimulus_move_before_goCue__lt,0.9,'
        '~session__extended_qc___task_audio_pre_trial__lt,0.9')

    marked_pass = (
        'session__extended_qc___experimenter_task,PASS')

    insertions = list(one.alyx.rest('insertions', 'list', django=base_query + qc_pass))
    insertions.extend(list(one.alyx.rest('insertions', 'list', django=base_query + marked_pass))) #why?

    ll_df = pd.DataFrame({
        'pid': np.array([i['id'] for i in insertions]),
        'eid': np.array([i['session'] for i in insertions]),
        'probe_name': np.array([i['name'] for i in insertions]),
        'session_number': np.array([i['session_info']['number'] for i in insertions]),
        'date': np.array([parser.parse(i['session_info']['start_time']).date() for i in insertions]),
        'subject': np.array([i['session_info']['subject'] for i in insertions]),
        'lab': np.array([i['session_info']['lab'] for i in insertions]),
    }).sort_values(by=['lab', 'subject', 'date', 'eid'])
    ll_df.drop_duplicates(inplace=True)
    ll_df.reset_index(inplace=True, drop=True)

    if return_details:
        return ll_df, insertions
    else:
        return ll_df

