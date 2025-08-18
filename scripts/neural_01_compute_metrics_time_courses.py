"""
compute FF 

input:
output:

"""
#%%
import config as C
import logging
import traceback
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from one.api import ONE
from iblatlas.atlas import AllenAtlas
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.closed_loop import compute_comparison_statistics
from brainbox.io.one import SpikeSortingLoader
from iblutil.numerical import ismember
from iblatlas.regions import BrainRegions
from scripts.utils.neuron_utils import cal_presence_ratio, combine_regions, smoothing_sliding
from scripts.utils.behavior_utils import clean_rts, filter_trials
from glob import glob
import logging
from scripts.utils.io import read_table

def clean_rt_table(trials_table,rt_variable ):

    trials_table['rt_raw'] = trials_table[rt_variable].copy()
    trials_table['rt'] = clean_rts(trials_table[rt_variable], cutoff=C.RT_CUTOFF)    
    return trials_table

def enrich_df(conditionsplit=False, k=None, id=None, ff_residuals=None, fr_residuals=None, time=None, bins_residuals_mean=None, bins_residuals_var=None, 
                fr_normed=None,ff_normed=None,fr=None,ff=None,
              pid=None, eid=None, lab=None, age_at_recording=None, sex=None, subject=None, sess_date=None, signed_contrast=None,
              n_trials=None, trial_type=None,
              clusters=None, cluster_idx=None): #, pids_un_young=None 

    #for each neuron/cluster:
    curr = {}
    curr['FFs_residuals'] = ff_residuals[k,:]
    curr['FF_residuals_mean'] = bins_residuals_mean[k,:]
    curr['FF_residuals_variance'] = bins_residuals_var[k,:]
    curr['frs_normed'] = fr_normed[k,:]
    curr['FFs_normed'] = ff_normed[k,:]
    curr['frs'] = fr[k,:]
    curr['FFs'] = ff[k,:]
    curr['frs_residuals'] = fr_residuals[k,:]
    # curr['frs_normalized'] = fr_normalized[k,:] #TODO:
    curr['timepoints'] = time

    df_curr = pd.DataFrame.from_dict(curr) 

    #4.region
    #5.cluster id
    df_curr['cluster_ids'] = id
    df_curr['uuids'] = clusters['uuids'][cluster_idx].iloc[k]
    df_curr['cluster_region'] = clusters['Beryl_merge'][cluster_idx][k] 
    df_curr['presence_ratio'] = clusters['presence_ratio'][cluster_idx][k]
    df_curr['contamination'] = clusters['contamination'][cluster_idx][k]

    df_curr['presence_ratio_std'] = clusters['presence_ratio_std'][cluster_idx][k]

    df_curr['firing_rate'] = clusters['firing_rate'][cluster_idx][k] 
    df_curr['presence_ratio_poi'] = clusters['presence_ratio_poi'][cluster_idx][k]
    df_curr['firing_rate_poi'] = clusters['firing_rate_poi'][cluster_idx][k] 
    df_curr['x'] = clusters['x'][cluster_idx][k] 
    df_curr['y'] = clusters['y'][cluster_idx][k] 
    df_curr['z'] = clusters['z'][cluster_idx][k]   
    
    
    ##those clusters share the same info:
    #1.group

    #2.mouse
    df_curr['mouse_age'] = age_at_recording
    df_curr['mouse_sex'] = sex
    df_curr['mouse_sub_name'] = subject
    df_curr['mouse_lab']= lab
    #3.session
    df_curr['session_eid'] = eid
    df_curr['session_pid'] = pid
    df_curr['session_date'] = sess_date

    # #6.trials
    # if conditionsplit:
    #     df_curr['signed_contrast'] = signed_contrast

    df_curr['n_trials']=n_trials  
    df_curr['trials_type']= C.TRIAL_TYPE

    return df_curr

def enrich_df_conditions(conditionsplit=True, k=None, id=None, ff=None, fr=None, fr_normed=None,ff_normed=None, time=None, ff_mean=None, ff_var=None, 
              pid=None, eid=None, lab=None, age_at_recording=None, sex=None, subject=None, sess_date=None, 
              n_trials=None, trial_type=None, signed_contrast=None,
              clusters=None, cluster_idx=None): #, pids_un_young=None 

    #for each neuron/cluster:
    curr = {}
    curr['FFs'] = ff[k,:]
    curr['FFs_normed'] = ff_normed[k,:]
    curr['FF_mean'] = ff_mean[k,:]
    curr['FF_variance'] = ff_var[k,:]
    curr['frs'] = fr[k,:]

    curr['frs_normed'] = fr_normed[k,:] 
    curr['timepoints'] = time

    df_curr = pd.DataFrame.from_dict(curr) 

    #4.region
    #5.cluster id
    df_curr['cluster_ids'] = id
    df_curr['uuids'] = clusters['uuids'][cluster_idx].iloc[k]
    df_curr['cluster_region'] = clusters['Beryl_merge'][cluster_idx][k] 
    df_curr['presence_ratio'] = clusters['presence_ratio'][cluster_idx][k]
    df_curr['contamination'] = clusters['contamination'][cluster_idx][k]

    df_curr['presence_ratio_std'] = clusters['presence_ratio_std'][cluster_idx][k]

    df_curr['firing_rate'] = clusters['firing_rate'][cluster_idx][k] 
    df_curr['presence_ratio_poi'] = clusters['presence_ratio_poi'][cluster_idx][k]
    df_curr['firing_rate_poi'] = clusters['firing_rate_poi'][cluster_idx][k] 
    df_curr['x'] = clusters['x'][cluster_idx][k] 
    df_curr['y'] = clusters['y'][cluster_idx][k] 
    df_curr['z'] = clusters['z'][cluster_idx][k]   
    
    ##those clusters share the same info:
    #1.group
    # if pid in pids_un_young:
    #     df_curr['age_group'] = 'young'
    # else:
    #     df_curr['age_group'] = 'old'

    #2.mouse
    df_curr['mouse_age'] = age_at_recording
    df_curr['mouse_sex'] = sex
    df_curr['mouse_sub_name'] = subject
    df_curr['mouse_lab']= lab
    #3.session
    df_curr['session_eid'] = eid
    df_curr['session_pid'] = pid
    df_curr['session_date'] = sess_date

    #6.trials
    # if conditionsplit:
    df_curr['signed_contrast'] = signed_contrast

    df_curr['n_trials']=n_trials  
    df_curr['trials_type']= C.TRIAL_TYPE

    return df_curr
def normalize_epoch_extremum(matrix_target, matrix_ref):
    """
    paras:
    matrix --  (trials, units, timepoints) array
    """
    trials, units, timepoints = matrix_target.shape
    result = np.zeros_like(matrix_target)
    for i in range(units):
        data = matrix_ref[:, i, :]
        data_across_trials =  np.mean(data, axis=0)  
        min_val = np.min(data_across_trials)
        max_val = np.max(data_across_trials)
        if max_val > min_val:
            denom = max(max_val - min_val, 1e-6)
            result[:, i, :] = (matrix_target[:, i, :] - min_val) / denom

        else:
            result[:, i, :] = np.zeros_like(matrix_target[:, i, :]) 
    return result

def extract_mouse_info (trials_table, eid):
    #extract trials and subject info from trials table
    trials = trials_table.loc[trials_table['eid']==eid] 
    subject = trials_table[trials_table['eid'] == eid]['mouse_name'].iloc[0]
    sex = trials_table[trials_table['eid'] == eid]['mouse_sex'].iloc[0]
    age_at_recording=  trials_table[trials_table['eid'] == eid]['mouse_age'].iloc[0]
    sess_date = trials_table[trials_table['eid'] == eid]['date'].iloc[0]

    return trials, subject, sex, age_at_recording, sess_date

def map_event():
    if C.ALIGN_EVENT == 'stim': #'stim','move'
        event = 'stimOn_times' #movement, feedback
    elif C.ALIGN_EVENT == 'move':
        event = 'firstMovement_times' #movement, feedback
    elif C.ALIGN_EVENT == 'feedback':
        event = 'feedback_times' #movement, feedback
    return event

def load_and_prepare_trials( trial_type, event_list, clean_rt, rt_variable_name):
    """
    Load and filter trials table.
    Returns: cleaned trials_table
    """
    trials_path = os.path.join(C.DATAPATH, 'ibl_included_eids_trials_table2025_full.csv')
    try:
        trials_table = pd.read_csv(trials_path)
    except Exception as err:
        print(f'Error loading trials table: {err}')
        return None

    if event_list:
        trials_table['exclude_nan_event_mask'] = np.where(trials_table[event_list].notna().all(axis=1), 1, 0)
        trials_table = trials_table[trials_table['exclude_nan_event_mask'] == 1]

    if trial_type == 'first400':
        trials_table = trials_table.groupby('eid').head(400).reset_index(drop=True)

    if clean_rt:
        trials_table = clean_rt_table(trials_table, rt_variable_name)
        trials_table = trials_table[~trials_table['rt'].isna()]
    
    return trials_table


def load_sorting_and_clusters(row, one, no_iblsortor):
    """
    Load spikes, clusters, and channels for a given row.
    Updates and returns no_iblsortor list if fallback is used.
    """
    pid = row['pid']
    if row['project'] == 'brainwidemap':
        sl = SpikeSortingLoader(one=one, pid=pid)
        spikes, clusters, channels = sl.load_spike_sorting(revision='2024-05-06', good_units=True)
        clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels, compute_metrics=False)

    elif row['project'] == 'learninglifespan':
        sl = SpikeSortingLoader(pid=pid, one=one, spike_sorter='iblsort')
        try:
            spikes, clusters, channels = sl.load_spike_sorting(enforce_version=True)
        except AssertionError:
            spikes, clusters, channels = sl.load_spike_sorting(enforce_version=False)
            no_iblsortor.append(pid)
        clusters = sl.merge_clusters(spikes, clusters, channels)

    return spikes, clusters, channels, no_iblsortor


def compute_cluster_metrics(spikes, clusters, trials, br, hist_win=10):
    """
    Compute presence ratio and firing rate for clusters within trial-aligned window.
    Also maps brain regions to Beryl and merged region.

    Returns:
        clusters: updated DataFrame with metrics and region labels
        spike_times_btw: filtered spike times (within window)
        spike_clusters: corresponding spike cluster ids
    """
    event = map_event()
    start_point = trials[event].min() - 0.4
    end_point = trials[event].max() + 1

    mask = (spikes['times'] >= start_point) & (spikes['times'] <= end_point)
    spike_times_btw = spikes['times'][mask]
    spike_clusters = spikes['clusters'][mask]
    cluster_ids = clusters['cluster_id']

    pr_poi, fr_poi = cal_presence_ratio(start_point, end_point, spike_times_btw, spike_clusters, cluster_ids, hist_win=hist_win)
    clusters['presence_ratio_poi'] = pr_poi
    clusters['firing_rate_poi'] = fr_poi

    clusters['Beryl'] = br.id2acronym(clusters['atlas_id'], mapping='Beryl')
    clusters['Beryl_merge'] = combine_regions(clusters['Beryl'])

    return clusters, spike_times_btw, spike_clusters


def compute_neural_yield(clusters, channels, ROIs, firing_rate_threshold, presence_ratio_threshold, pid, eid, subject, age_at_recording, br):
    """
    Given clusters and channels with metrics and region info,
    compute the number of good units and channels per brain region (yield).
    
    Returns:
        yield_table: DataFrame with n_channel, n_cluster, and metadata
        clusters_ids: selected good cluster IDs
        cluster_idx: boolean mask of selected clusters
    """
    # Handle channels
    channels['Beryl'] = br.id2acronym(channels['atlas_id'], mapping='Beryl')
    channels['Beryl_merge'] = combine_regions(channels['Beryl'])

    try:
        channels_df = pd.DataFrame.from_dict(channels)
    except Exception as err:
        print(f"Error creating channels_df, falling back: {err}")
        for key, value in channels.items():
            print(f"{key}: {len(value)}")

        selected_columns = ['x', 'y', 'z', 'acronym', 'atlas_id', 'Beryl', 'Beryl_merge']
        filtered_data = {key: channels[key] for key in selected_columns if key in channels}
        channels_df = pd.DataFrame(filtered_data)

    channels_good = (
        channels_df[channels_df["Beryl_merge"].isin(C.ROIS)]
        .groupby("Beryl_merge")[["rawInd"]]
        .nunique()
        .reset_index()
        .rename(columns={"rawInd": "n_channel"})
    )

    # Select good clusters
    cluster_idx = (
        np.isin(clusters['Beryl_merge'], ROIs) &
        (clusters['label'] == 1) &
        (clusters['firing_rate_poi'] > firing_rate_threshold) &
        (clusters['presence_ratio_poi'] > presence_ratio_threshold)
    )
    clusters_ids = clusters['cluster_id'][cluster_idx]
    clusters_df = pd.DataFrame.from_dict(clusters)

    cluster_good = (
        clusters_df.loc[cluster_idx]
        .groupby("Beryl_merge")["cluster_id"]
        .nunique()
        .reset_index()
        .rename(columns={"cluster_id": "n_cluster"})
    )

    # Merge channel and cluster counts
    yield_table = pd.merge(channels_good, cluster_good, on="Beryl_merge", how="outer").reset_index(drop=True)
    yield_table['pid'] = pid
    yield_table['eid'] = eid
    yield_table['subject'] = subject
    yield_table['age_at_recording'] = age_at_recording

    return yield_table, clusters_ids, cluster_idx


def filter_spikes_by_cluster(spikes, clusters_ids, pid, pid_no_spikes):
    """
    Filters spikes based on selected cluster IDs.

    Returns:
        spike_idx: boolean mask
        has_valid_spikes: bool
        updated pid_no_spikes (optional)
    """
    spike_idx = np.isin(spikes['clusters'], clusters_ids)
    if np.sum(spike_idx) == 0:
        print(f"{pid} — No spikes in selected C.ROIS.")
        pid_no_spikes.append(pid)
        return spike_idx, False, pid_no_spikes
    return spike_idx, True, pid_no_spikes


def compute_fano_factors(spikes, spike_idx, clusters_ids, trials, event, event_epoch, bin_size, 
                         eid, pid, age_at_recording, sex, subject, sess_date, trial_type,
                         clusters, cluster_idx):
    """
    Compute both per-condition FF and residual (mean-subtracted) FF for each neuron.

    Returns:
        df_all: list of enriched df for residual FF (enrich_df)
        df_all_conditions: list of enriched df for each condition (enrich_df_conditions)
    """
    df_all = []
    df_all_conditions = []
    all_sc_adjusted_bins = []
    weighted_mean_sum = None
    total_weight = 0

    # full bins: all trials
    bins_full, t_full = smoothing_sliding(
        spikes['times'][spike_idx], spikes['clusters'][spike_idx], clusters_ids,
        trials[event].values, align_epoch=event_epoch, bin_size=bin_size
    )

    # compute condition-specific FF
    for signed_contrast, group in trials.groupby('signed_contrast'):
        bins, t = smoothing_sliding(
            spikes['times'][spike_idx], spikes['clusters'][spike_idx], clusters_ids,
            group[event].values, align_epoch=event_epoch, bin_size=bin_size
        )
        num_trials = bins.shape[0]

        sc_normalized = normalize_epoch_extremum(bins, bins_full)
        fr_normalized = normalize_epoch_extremum(bins/bin_size, bins_full/bin_size)
        fr_mean_across_trials = np.mean(bins/bin_size, axis=0)
        fr_normalized_mean_across_trials = np.mean(fr_normalized, axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            ff = np.nanvar(bins, axis=0) / np.nanmean(bins, axis=0)

        ff_var = np.nanvar(bins, axis=0)
        ff_mean = np.nanmean(bins, axis=0)
        ff_normed = np.nanvar(sc_normalized, axis=0) / np.nanmean(sc_normalized, axis=0)

        sc_mean_across_trials = np.mean(bins, axis=0)
        if weighted_mean_sum is None:
            weighted_mean_sum = sc_mean_across_trials * num_trials
        else:
            weighted_mean_sum += sc_mean_across_trials * num_trials
        total_weight += num_trials

        sc_adjusted_bins = bins - sc_mean_across_trials
        all_sc_adjusted_bins.append(sc_adjusted_bins)

        for k, id in enumerate(clusters_ids):
            df_curr_conditions = enrich_df_conditions(
                conditionsplit=True, k=k, id=id,
                ff=ff, ff_normed=ff_normed, fr=fr_mean_across_trials, fr_normed=fr_normalized_mean_across_trials,
                ff_mean=ff_mean, ff_var=ff_var, time=t, signed_contrast=signed_contrast,
                eid=eid, pid=pid, age_at_recording=age_at_recording, sex=sex, subject=subject,
                sess_date=sess_date, n_trials=num_trials, trial_type=trial_type,
                clusters=clusters, cluster_idx=cluster_idx
            )
            df_all_conditions.append(df_curr_conditions)

    # residual FF (pooled over all adjusted bins)
    concatenated_bins = np.concatenate(all_sc_adjusted_bins, axis=0)
    weighted_mean_across_conditions = weighted_mean_sum / total_weight
    ff_residuals = np.nanvar(concatenated_bins, axis=0) / weighted_mean_across_conditions
    fr_residuals = weighted_mean_across_conditions / bin_size
    ff_var_residuals = np.nanvar(concatenated_bins, axis=0)
    ff_mean_residuals = weighted_mean_across_conditions
    ff_time = t

    # full bins normalized FF
    sc_normalized_full = normalize_epoch_extremum(bins_full, bins_full)
    fr_normalized_full_mean_across_trials = np.mean(sc_normalized_full, axis=0)
    ff_full_normed = np.nanvar(sc_normalized_full, axis=0) / np.nanmean(sc_normalized_full, axis=0)
    ff_full_original = np.nanvar(bins_full, axis=0) / np.nanmean(bins_full, axis=0)
    fr_full_original = np.mean(bins_full / bin_size, axis=0)

    for k, id in enumerate(clusters_ids):
        df_curr = enrich_df(
            conditionsplit=False, k=k, id=id,
            ff_residuals=ff_residuals, fr_residuals=fr_residuals,
            ff=ff_full_original, fr=fr_full_original,
            fr_normed=fr_normalized_full_mean_across_trials, ff_normed=ff_full_normed,
            time=ff_time, bins_residuals_mean=ff_mean_residuals, bins_residuals_var=ff_var_residuals,
            eid=eid, pid=pid, age_at_recording=age_at_recording, sex=sex, subject=subject,
            sess_date=sess_date, n_trials=total_weight, trial_type=trial_type,
            clusters=clusters, cluster_idx=cluster_idx
        )
        df_all.append(df_curr)

    return df_all, df_all_conditions


def save_results_to_parquet(df_all, df_conditions, neural_yield, suffix="2025"):
    """
    Save result DataFrames to parquet files with standardized naming.
    """
    ff_file = C.DATAPATH / f'ibl_BWMLL_FFs_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_{suffix}.parquet'
    cond_file = C.DATAPATH / f'ibl_BWMLL_FFs_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_conditions_{suffix}.parquet'
    yield_file = C.DATAPATH / f'ibl_BWMLL_neural_yield_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_{suffix}.parquet'

    df_all.to_parquet(ff_file, engine='pyarrow', compression='snappy')
    df_conditions.to_parquet(cond_file, engine='pyarrow', compression='snappy')
    neural_yield.to_parquet( yield_file, engine='pyarrow', compression='snappy')

    print(" Results saved successfully:")
    print(f"    • FF (overall):     {ff_file}")
    print(f"    • FF (conditions):  {cond_file}")
    print(f"    • Neural yield:     {yield_file}")



log_file_path = C.DATAPATH / f"error_log_FF_compute_{C.ALIGN_EVENT}_{C.TRIAL_TYPE}.log"

logging.basicConfig(
    filename=log_file_path,
    filemode='w',  
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
processed_dir = C.DATAPATH / "processed_pids"
processed_dir.mkdir(exist_ok=True)


if __name__ == "__main__":
    
    save_results = True
    mean_subtraction = False
    clean_rt = True 

    trials_table = load_and_prepare_trials(C.TRIAL_TYPE, C.EVENT_LIST, C.CLEAN_RT, C.RT_VARIABLE_NAME)
    
    if trials_table is None:
        print("Failed to load trials.")

    print(len(set(trials_table.eid)))

    recordings_filtered = read_table(C.DATAPATH / 'BWM_LL_release_afterQC_df.csv')
    #TODO: 为啥去掉了3个pids?
    # recordings_filtered = recordings_filtered[~recordings_filtered['pid'].isin(notget_pids)]

    ba = AllenAtlas()
    br = BrainRegions()
    one = ONE()
    no_iblsortor=[]
    pid_no_spikes=[]

    list_ind = []
    all_pids_to_use = recordings_filtered['pid'].to_list()
    for index, row in recordings_filtered.iterrows():
        #TODO: 确认之后，去掉那4个？
        pid = row['pid']
        if pid in ['57edc590-a53d-403c-9aab-d58ee51b6a24', 'daadb3f1-bef2-474e-a659-72922f3fcc5b', '61bb2bcd-37b4-4bcc-8f40-8681009a511a', 'ee2ce090-696a-40f5-8f29-7107339bf08e']:
            continue
        
        print(f"PID {index+1}/{len(recordings_filtered)}: {pid}")
        print(pid)

        done_file = processed_dir / f"{pid}.done"
        if done_file.exists():
            print(f"Skipping already processed pid: {pid}")
            continue

        print(f'processing {index + 1}/{len(recordings_filtered)}') #

        try:
            list_ind.append(index)
            eid, pname = one.pid2eid(pid)
            eid = str(eid)
            trials, subject, sex, age_at_recording, sess_date = extract_mouse_info(trials_table, eid)
            spikes, clusters, channels, no_iblsortor = load_sorting_and_clusters(row, one, no_iblsortor)

            event = map_event()

            clusters, spike_times_btw, spike_clusters = compute_cluster_metrics(spikes, clusters, trials, br)


            yield_table, clusters_ids, cluster_idx = compute_neural_yield(
                                clusters, channels, C.ROIS, C.FIRING_RATE_THRESHOLD,
                                C.PRESENCE_RATIO_THRESHOLD, pid, eid, subject, age_at_recording, br
                            )
            # neural_yield_all.append(yield_table)  
            spike_idx, has_valid_spikes, pid_no_spikes = filter_spikes_by_cluster(spikes, clusters_ids, pid, pid_no_spikes)
            if not has_valid_spikes:
                continue


            df_this, df_conditions_this = compute_fano_factors(
                spikes, spike_idx, clusters_ids, trials, event, C.EVENT_EPOCH, C.BIN_SIZE,
                eid, pid, age_at_recording, sex, subject, sess_date, C.TRIAL_TYPE,
                clusters, cluster_idx
            )
            # df_all.extend(df_this)
            # df_all_conditions.extend(df_conditions_this)

            #2. save results for current pid
            df_pid = pd.concat(df_this, ignore_index=True)
            df_cond_pid = pd.concat(df_conditions_this, ignore_index=True)

            df_pid.to_parquet(processed_dir / f"{pid}_ff.parquet", engine='pyarrow', compression='snappy')
            df_cond_pid.to_parquet(processed_dir / f"{pid}_ff_cond.parquet", engine='pyarrow', compression='snappy')
            yield_table.to_parquet(processed_dir / f"{pid}_yield.parquet", engine='pyarrow', compression='snappy')

            # 3. mark it with a .done file
            done_file.touch()

        except Exception as err:
            print(f"Error on PID {pid} (index {index}): {err}")
            traceback.print_exc()

            logging.error(f"PID {pid} (index {index}) — {err}")
            logging.error(traceback.format_exc())

            continue

# all pids expected
expected_pids = set(recordings_filtered['pid'])

# pids processed, with a .done file
done_pids = {Path(f).stem for f in glob(str(processed_dir / "*.done"))}

# any missing pid
missing_pids = expected_pids - done_pids

if missing_pids:
    print(f"Warning: {len(missing_pids)} PIDs missing results.")
    # print(f"Missing PIDs (sample): {list(missing_pids)[:5]}")
else:
    print("All expected PIDs are present.")

df_all = pd.concat([read_table(f) for f in glob(str(processed_dir / "*_ff.parquet"))], ignore_index=True)
df_all_conditions = pd.concat([read_table(f) for f in glob(str(processed_dir / "*_ff_cond.parquet"))], ignore_index=True)
neural_yield_all = pd.concat([read_table(f) for f in glob(str(processed_dir / "*_yield.parquet"))], ignore_index=True)

# 5. merge and save results
if save_results:
    save_results_to_parquet(
        df_all, df_all_conditions, neural_yield_all, suffix="2025")

# %%
