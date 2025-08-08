"""
data release
double check QUERY FUNCTION

"""

#%%
import json
from dateutil import parser
import numpy as np
import pandas as pd
from pathlib import Path
import os

from iblutil.numerical import ismember
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from brainbox.behavior import training
from iblatlas.regions import BrainRegions
# from ibllib.qc.base import CRITERIA
# from one.remote import aws
# import brainwidemap


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
        '~session__json__IS_MOCK,True,' #TODO: what does this mean?
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

#%%
from one.api import ONE

one = ONE()
lifespan_df = lifespan_query(one=one, alignment_resolved=True, return_details=False)
print(len(lifespan_df)) #68 

#%%
pidlist2print = lifespan_df.pid.to_list()
print(pidlist2print) #68
# ['aebf71d2-408d-4901-a4e3-007cf61af8a0', '3ae57f3f-be93-48d9-8f47-f5a6f9055a3a', '34d7572f-7e3b-4d12-afc5-c26bd2dfde15', 'fa0dc413-1525-4178-89a8-3a093e08e2ce', '14e496f3-06c7-465e-b7a7-109d866793a2', '96bfe790-37a6-45da-aa30-e271c9ce68e6', '35e215e1-b925-4593-8d67-0de6603c525a', 'c7b21911-7699-402c-badc-670f723a7e42', '58153f9d-08b5-4386-b1ad-715c3922a470', '7d5c4c4e-5c23-4a1d-a1ce-32cb79bbcc2a', '2631e77a-d521-4939-b247-e7a0ea0a95c1', 'bf8bb47d-e7e4-4759-87f1-fe054407661b', '32622302-ffef-412b-8773-4c7ef2a993bf', '42fd1bb1-76b1-4227-9a74-925bcd28b9a0', '383e12dc-4825-46f4-ac1d-8f3f6a6a1381', 'cbe746bb-8076-41c9-a90e-3bd56b2d958e', '3b7827f2-3957-41dd-8b12-fa529a1422e7', '278ce01a-2383-4a09-b74a-b8571ccbabad', '5ce945a6-5b59-4d30-8be9-51f6e8280b43', '85b98361-9706-4318-8923-6988d4e804e8', 'ba40eda8-601d-41cb-a629-290d17e7a680', '23e12f60-05fc-43ce-9536-f2feef8db037', '39e6c9a9-2241-4781-9ed6-db45979207e7', '3d9db0eb-31a6-44a8-99de-6e04555d27be', '797fd358-5778-4b9f-b037-2aeb2393b839', 'd2da187c-7277-4114-bd09-f4f62ce9947e', 'e096bf1f-b2b7-471d-b07b-e4e6d65299ac', '08c49305-d12c-4cc5-8f5c-b29f62f3a4a6', '7246e8f8-f694-488f-8ce8-5214975ffe9a', '646bfb77-b784-4c21-b37f-2ffc9986a228', 'f8965106-8f4a-4910-81f6-f19d55878b4e', '57edc590-a53d-403c-9aab-d58ee51b6a24', '9341b8f3-fa57-4fbb-9d0b-7ca3613da0cc', 'bb4fbbf0-4d1e-4d0d-b348-2d7b7fddd151', '18e665b6-cc3d-4cde-980b-ddba405c1b26', '4b345c19-4973-4f30-8858-f236e7456553', '770470e7-9e8c-40c0-b95b-33330c096ade', 'b9292b9f-cc04-4d2b-93c0-c0ad16e2b221', '6c74c0f6-030d-4665-9e2e-799b1bcd3367', 'becce8b9-db96-4ace-ad99-66397ca9e181', 'a7919b08-68fd-4ae7-a5a6-341d054d5bed', 'e8f25a3a-ab3d-4e3e-a863-5014a2b7e440', '556b57db-e4bd-456b-b0bf-d0ddc56603ff', '68386da9-4a3e-425f-baa1-15e4f985153d', '253768e5-b649-4e49-944d-cc79d30b8f35', '3c89bf07-1010-4457-8018-1733d50725e7', '488d4bf8-9ae6-4bbe-9025-fcfdaec93efd', '7f788d36-56dd-4ebc-863e-c22ec4f1a731', '429747bb-93c2-4ec2-b823-a49d9247d4d5', 'daadb3f1-bef2-474e-a659-72922f3fcc5b', '9b52d705-f975-4efb-863f-8c0ee33495c9', 'eb48b9fc-661e-4dde-8388-f32bde00482f', '4dd3d15f-b59b-4403-85c0-417d41337f5d', '7215243f-f336-4602-9521-6f9786d4decd', '8f4213bd-49ab-4421-bf1b-a1f8e6e1f37f', '02106173-e888-429b-a03c-7daaa40cc6be', '7ca1a6a0-5808-4882-9f8c-ea5ae82ad1a2', 'fc93dc34-3329-475b-b779-1167482c86b7', '0bc568eb-feed-4d5d-86e2-edfdf84ed707', 'feb4c65d-ad57-430e-8698-31f193013d19', '3adebebc-6499-4ad7-81db-a7a61f50fb15', '442d6f32-f0dc-4f82-90e3-5eefb086797c', 'c52c4943-e764-4a9d-a759-06aff36993f0', 'd7c474c8-168a-4ae3-a2d8-573ca8017708', '1edd8ae4-02bb-47ab-868c-1d5fad6256aa', '61bb2bcd-37b4-4bcc-8f40-8681009a511a', 'a089219a-c836-470a-91c7-d65617bfb82a', 'ee2ce090-696a-40f5-8f29-7107339bf08e']

eidlist2print = list(lifespan_df.eid.unique())
print(eidlist2print) #38
# ['89e258e9-cbca-4eca-bac4-13a2388b5113', 'c875fc7d-0966-448a-813d-663088fbfae8', '0fe99726-9982-4c41-a07c-2cd7af6a6733', 'c94463ed-57da-4f02-8406-46f2f03924f3', 'ba7fc4d0-0486-4415-9b12-3f13b1cff710', '93374502-c701-4b83-aa1a-23050b514708', '78fceb60-e623-431b-ab80-7e29209058ac', '7ae3865a-d8f4-4b73-938e-ddaec33f8bc6', '107249ca-0d03-4e56-a7eb-6fe6210550ae', 'ad8e802d-ce83-437a-865f-fa769762a602', '804bc680-976b-4e3e-9a47-a7e94847bd06', 'a0dfbbc6-0454-4dc6-ade0-9ba57c18241d', '2cff323c-1510-4b78-a5d1-ca07b203f60c', 'a5145869-a54a-4871-95ef-016421122844', 'f2545193-1c5c-420e-96ac-3cb4b9799ea5', 'ab8a5331-1d0f-4b8a-9e0f-7be41c4857f9', '9b4f6a8d-c879-4348-aa7e-0b34f6c6dacb', '022dd14c-eff2-470f-863c-e019fafa53ae', 'a06189b0-a66e-4a5a-a1ef-4afa80de8b31', 'bf358c9a-ef84-4604-b83a-93416d2827ff', '87b628a4-f11a-429c-ad98-34d43cf3178b', '945028b5-bb38-4379-8ae4-488bcd67bcf5', '5c936319-6829-41cb-abc7-c4430910a6a0', '531e7ac0-cfcd-4593-9bf7-bb7bab5d66e9', '3a1b819b-71ef-4d71-aae6-9f83c1f509cb', '11cc0294-fbc5-44b7-8a2c-484daa64c81e', 'fe0ecca9-9279-4ce6-bbfe-8b875d30d34b', 'b26295df-e78d-4368-b694-1bf584f25bfc', 'a45e62df-9f7f-4429-95a4-c4e334c8209f', 'fe80df7d-15f0-4f89-9bbb-d3e5725c4b0a', 'da9eeafc-d7af-4a19-bf1c-2064e5b1b696', 'bb2153e7-1052-491e-a022-790e755c7a54', 'f45e30cf-12aa-4fa0-8248-f9f885dfa9ef', 'c90cdfa0-2945-4f68-8351-cb964c258725', '9931191e-8056-4adc-a410-a4a93487423f', 'af74b29d-a671-4c22-a5e8-1e3d27e362f3', 'f31752a8-a6bb-498b-8118-6339d3d74ecb', '6f321eab-6dad-4f2e-8160-5b182f999bb6']


#%%camera QC
import json
from one.alf import spec
def filter_video_data(one, eids, camera='left', min_video_qc='FAIL', min_dlc_qc='FAIL'):
    """
    Filters sessions for which video and/or DLC data passes a given QC threshold for a selected camera.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to local or remote database.
    eids: list or str
        List of session UUIDs to filter.
    camera: {'left', 'right', 'body'}
        Camera for which to filter QC. Default is 'left'.
    min_video_qc: {'CRITICAL', 'FAIL', 'WARNING', 'PASS', 'NOT_SET'} or None
        Minimum video QC threshold for a session to be retained. Default is 'FAIL'.
    min_dlc_qc: {'CRITICAL', 'FAIL', 'WARNING', 'PASS', 'NOT_SET'} or None
        Minimum dlc QC threshold for a session to be retained. Default is 'FAIL'.

    Returns
    -------
    list:
        List of session UUIDs that pass both indicated QC thresholds for the selected camera.

    Notes
    -----
    For the thresholds, note that 'NOT_SET' < 'PASS' < 'WARNING' < 'FAIL' < 'CRITICAL'
    If a min_video_qc or min_dlc_qc is set to None, all sessions are retained for that criterion.
    The intersection of sessions passing both criteria is returned.

    """

    # Check inputs
    if isinstance(eids, str):
        eids = [eids]
    assert isinstance(eids, list), 'eids must be a list of session uuids'

    # Load QC json from cache and restrict to desired sessions
    # with open(one.cache_dir.joinpath('QC.json'), 'r') as f:
    #     qc_cache = json.load(f)
    # qc_list = one.get_details(eids, True)

    qc_cache = {eid: one.get_details(eid, True)['extended_qc'] for eid in eids}
    assert set(list(qc_cache.keys())) == set(eids), 'Not all eids found in cached QC.json'

    # Passing video
    if min_video_qc is None:
        passing_vid = eids
    else:
        passing_vid = [
            k for k, v in qc_cache.items() if
            f'video{camera.capitalize()}' in v.keys() and
            spec.QC[v[f'video{camera.capitalize()}']].value <= spec.QC[min_video_qc].value
        ]

    # Passing dlc
    if min_dlc_qc is None:
        passing_dlc = eids
    else:
        passing_dlc = [
            k for k, v in qc_cache.items() if
            f'dlc{camera.capitalize()}' in v.keys() and
            spec.QC[v[f'dlc{camera.capitalize()}']].value <= spec.QC[min_dlc_qc].value
        ]

    # Combine
    passing = list(set(passing_vid).intersection(set(passing_dlc)))
    return passing

videoQC_filter_left = filter_video_data(one, eidlist2print, camera='left', min_video_qc='FAIL', min_dlc_qc='FAIL') #324
videoQC_filter_body = filter_video_data(one, eidlist2print, camera='body', min_video_qc='FAIL', min_dlc_qc='FAIL') #192
videoQC_filter_both = list(set(videoQC_filter_left).intersection(set(videoQC_filter_body))) #192

#%%
res_list = []
for eid in eidlist2print:
    try:
        check_eid_extend_qc = one.get_details(eid, True)['extended_qc']
        # print(check_eid_extend_qc['dlcLeft'])
        # print(check_eid_extend_qc['videoLeft'])
        try:
            dlcLeft = check_eid_extend_qc['dlcLeft']
        except:
            dlcLeft = None
        try:
            videoLeft = check_eid_extend_qc['videoLeft']
        except:
            videoLeft = None
        try:
            dlcBody = check_eid_extend_qc['dlcBody']
        except:
            dlcBody = None
        try:
            videoBody = check_eid_extend_qc['videoBody']
        except:
            videoBody = None
        res_dict = {'eid':eid, 'dlcLeft':dlcLeft, 'videoLeft':videoLeft,'dlcBody':dlcBody, 'videoBody':videoBody}
        res_list.append(res_dict)
    except Exception as err:
        print(err)
        continue

res_df = pd.DataFrame(res_list)
#%%==============================================================
#     **check_wheel_camera_data_availablity.py**
# ===================================================================

for index, row in res_df.iterrows():
    datasets_all = one.list_datasets(row['eid'], collection='alf')

    result = any('ibl_wheel' in item for item in datasets_all)
    res_df.loc[index,'ibl_wheel'] = result
    
    result2 = any('leftCamera.ROIMotionEnergy' in item for item in datasets_all)
    res_df.loc[index,'left_camera'] = result2
    
    result3 = any('rightCamera.ROIMotionEnergy' in item for item in datasets_all)
    res_df.loc[index,'right_camera'] = result3

    result4 = any('bodyCamera.ROIMotionEnergy' in item for item in datasets_all)
    res_df.loc[index,'body_camera'] = result4    

res_df['n_camera'] = res_df['left_camera']+res_df['right_camera']+res_df['body_camera']
res_df = res_df.sort_values(by=['n_camera', 'ibl_wheel'], ascending=[True, True])

#%%





























#%%===========================================================
#                        test each step seperately: 
#=============================================================

base_query = (
    'session__projects__name,churchland_learninglifespan,'#112
    '~session__json__IS_MOCK,True,' #112
    'session__qc__lt,50,'# #112
    'session__extended_qc__behavior,1,'#75
    '~json__qc,CRITICAL,' #69
    'json__extended_qc__tracing_exists,True,'#69
    'json__extended_qc__alignment_resolved,True,'# 68
)

# base_only_ins2 = list(one.alyx.rest('insertions', 'list', django=base_query))
# print(len(base_only_ins2))#68

# ll_df2 = pd.DataFrame({
#     'pid': np.array([i['id'] for i in base_only_ins2]),
#     'eid': np.array([i['session'] for i in base_only_ins2]),
#     'probe_name': np.array([i['name'] for i in base_only_ins2]),
#     'session_number': np.array([i['session_info']['number'] for i in base_only_ins2]),
#     'date': np.array([parser.parse(i['session_info']['start_time']).date() for i in base_only_ins2]),
#     'subject': np.array([i['session_info']['subject'] for i in base_only_ins2]),
#     'lab': np.array([i['session_info']['lab'] for i in base_only_ins2]),
# }).sort_values(by=['lab', 'subject', 'date', 'eid'])
# ll_df2.drop_duplicates(inplace=True)
# ll_df2.reset_index(inplace=True, drop=True)
# print(ll_df2.eid.nunique()) #behavior 38; 
# print(ll_df2.pid.nunique()) 


# if alignment_resolved:
#     base_query += 'json__extended_qc__alignment_resolved,True,'
# else:
#     base_query += 'json__extended_qc__alignment_count__gt,0,'

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
print(len(insertions))#68
#%%
extra = list(one.alyx.rest('insertions', 'list', django=base_query + marked_pass))
print(len(extra)) #0
#%%
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