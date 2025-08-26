"""
Compute Fano Factors (FF) and related neural metrics.


This script orchestrates the pipeline to:
1) load trial tables and spike sorting outputs,
2) compute per-cluster metrics (presence ratio, firing rate) within a time window of interest,
3) select "good" clusters per brain region (yield),
4) compute FR/FF time courses per condition (signed contrast) and residual FF after mean subtraction across conditions,
5) persist per-pid intermediate results and finally merge and save tables.


Inputs (via config file and data files):
• C.DATAPATH / 'ibl_included_eids_trials_table2025_full.csv' — trials table with per‑trial metadata
• C.DATAPATH / 'BWM_LL_release_afterQC_df.csv' — filtered list of PIDs/EIDs to process
• spike sorting data — accessed with `SpikeSortingLoader` via ONE


Key configurable parameters (see config file):
• ALIGN_EVENT ('stim' | 'move')
• TRIAL_TYPE (e.g., 'first400')
• EVENT_LIST — required event columns that must be non-NaN to use a trial
• CLEAN_RT & RT_VARIABLE_NAME — reaction time cleaning
• BIN_SIZE — smoothing/analysis bin size (s)
• EVENT_EPOCH — time window around ALIGN_EVENT to align spikes
• ROIS — list of merged Beryl regions to include
• PRESENCE_RATIO_THRESHOLD, FIRING_RATE_THRESHOLD — cluster selection


Outputs (saved under `C.DATAPATH`):
• ibl_BWMLL_FFs_{ALIGN_EVENT}_{TRIAL_TYPE}_2025.parquet
• ibl_BWMLL_FFs_{ALIGN_EVENT}_{TRIAL_TYPE}_conditions_2025.parquet
• ibl_BWMLL_neural_yield_{ALIGN_EVENT}_{TRIAL_TYPE}_2025.parquet


Per pid intermediates (under `C.DATAPATH / processed_pids`):
• {pid}_ff.parquet — residual + full FF/FR per unit×timepoint
• {pid}_ff_cond.parquet — condition‑wise FF/FR per unit×timepoint
• {pid}_yield.parquet — channel/cluster counts per region + metadata
• {pid}.done — sentinel file marking successful processing


Notes:
• To resume safely, the main loop skips pids with an existing `.done` file.
"""
#%%
import config as C
import logging
import traceback
import os
import pandas as pd
import numpy as np
from pathlib import Path
from one.api import ONE
from iblatlas.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader
from iblutil.numerical import ismember
from iblatlas.regions import BrainRegions
from scripts.utils.neuron_utils import cal_presence_ratio, combine_regions, smoothing_sliding
from scripts.utils.behavior_utils import clean_rts
from glob import glob
import logging
from scripts.utils.io import read_table

log = logging.getLogger(__name__)

def clean_rt_table(trials_table, rt_variable):
    """Add cleaned rt to a trials table.


    Parameters
    ----------
    trials_table : pd.DataFrame
    Table containing trial level variables. Must include column `rt_variable`.
    rt_variable : str
    Name of the column with raw rt to be cleaned.


    Returns
    -------
    pd.DataFrame
    Same table with two extra columns:
    • 'rt_raw' — the original values (copy)
    • 'rt' — the cleaned RT with outliers removed using `C.RT_CUTOFF`
    """
    trials_table['rt_raw'] = trials_table[rt_variable].copy()
    trials_table['rt'] = clean_rts(trials_table[rt_variable], cutoff=C.RT_CUTOFF)    
    return trials_table


def enrich_df(conditionsplit=False, k=None, id=None, ff_residuals=None, fr_residuals=None, time=None, bins_residuals_mean=None, bins_residuals_var=None, 
                fr_normed=None,ff_normed=None,fr=None,ff=None,
              pid=None, eid=None, lab=None, age_at_recording=None, sex=None, subject=None, sess_date=None, signed_contrast=None,
              n_trials=None, trial_type=None,
              clusters=None, cluster_idx=None): 
    """Assemble a per-unit time-course row-wise table for residual/full FF and FR.


    This version is used for residual (mean-subtracted across conditions) summaries
    and full-trial summaries (not condition-split). One unit contributes len(time)
    rows (one per timepoint) to the output.


    Parameters (selected)
    ---------------------
    k : int
    Index of the unit within `clusters_ids` / `cluster_idx`.
    id : int
    Cluster id for the current unit.
    ff_residuals, fr_residuals : np.ndarray [n_units, n_time]
    Residual FF and FR time courses after mean subtraction across conditions.
    time : np.ndarray [n_time]
    Time axis (relative to ALIGN_EVENT).
    bins_residuals_mean, bins_residuals_var : np.ndarray [n_units, n_time]
    Mean/variance of the adjusted spike counts used to form FF residuals.
    fr_normed, ff_normed, fr, ff : np.ndarray [n_units, n_time]
    Normalized FR/FF and original FR/FF from full trials.
    clusters : dict-like
    Cluster table as returned/merged by `SpikeSortingLoader`.
    cluster_idx : boolean array
    Mask selecting the good units among `clusters`.


    Returns
    -------
    pd.DataFrame
    One unit × timepoint table with metrics + unit/session metadata.
    """
    # Per‑unit time‑series metrics (shape: n_time, each row one timepoint)
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
    # Unit identity and quality/geometry
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
    # Session / subject metadata
    df_curr['mouse_age'] = age_at_recording
    df_curr['mouse_sex'] = sex
    df_curr['mouse_sub_name'] = subject
    df_curr['mouse_lab']= lab
    df_curr['session_eid'] = eid
    df_curr['session_pid'] = pid
    df_curr['session_date'] = sess_date

    # Trial bookkeeping (not condition‑split here)
    df_curr['n_trials']=n_trials  
    df_curr['trials_type']= C.TRIAL_TYPE

    return df_curr


def enrich_df_conditions(conditionsplit=True, k=None, id=None, ff=None, fr=None, fr_normed=None,ff_normed=None, time=None, ff_mean=None, ff_var=None, 
              pid=None, eid=None, lab=None, age_at_recording=None, sex=None, subject=None, sess_date=None, 
              n_trials=None, trial_type=None, signed_contrast=None,
              clusters=None, cluster_idx=None): #, pids_un_young=None 
    """Assemble a per-unit time-course table for *condition-split* FF and FR.


    Each unit contributes len(time) rows per condition (signed_contrast).


    Returns
    -------
    pd.DataFrame
    One unit x timepoint x condition table with metrics + unit/session metadata.
    """
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

    # Unit identity and quality/geometry
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
    
    # Session / subject metadata
    df_curr['mouse_age'] = age_at_recording
    df_curr['mouse_sex'] = sex
    df_curr['mouse_sub_name'] = subject
    df_curr['mouse_lab']= lab
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
    """Min-max normalize a 3D tensor per unit using the reference epoch extrema.


    Parameters
    ----------
    matrix_target : np.ndarray [n_trials, n_units, n_time]
    Tensor to normalize.
    matrix_ref : np.ndarray [n_trials, n_units, n_time]
    Reference tensor used to compute per-unit min/max across trials and time.


    Returns
    -------
    np.ndarray
    Normalized tensor of the same shape in [0, 1]. Units with flat reference
    activity are set to zeros to avoid division by zero.
    """
    trials, units, timepoints = matrix_target.shape
    result = np.zeros_like(matrix_target)
    for i in range(units):
        # Aggregate across trials for unit i using the reference epoch
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


def extract_mouse_info(trials_table, eid):
    """Extract per-session subject metadata from the trials table.


    Parameters
    ----------
    trials_table : pd.DataFrame
    Trials table containing columns: 'eid', 'mouse_name', 'mouse_sex',
    'mouse_age', 'date'.
    eid : str
    Session EID.


    Returns
    -------
    (pd.DataFrame, str, str, float/int, str)
    (trials_subtable, subject, sex, age_at_recording, sess_date)
    """
    
    #extract trials and subject info from trials table
    trials = trials_table.loc[trials_table['eid']==eid] 
    subject = trials_table[trials_table['eid'] == eid]['mouse_name'].iloc[0]
    sex = trials_table[trials_table['eid'] == eid]['mouse_sex'].iloc[0]
    age_at_recording = trials_table[trials_table['eid'] == eid]['mouse_age'].iloc[0]
    sess_date = trials_table[trials_table['eid'] == eid]['date'].iloc[0]

    return trials, subject, sex, age_at_recording, sess_date


def map_event():
    """Map `C.ALIGN_EVENT` to the corresponding trials column name."""
    if C.ALIGN_EVENT == 'stim': #'stim','move'
        event = 'stimOn_times' #movement, feedback
    elif C.ALIGN_EVENT == 'move':
        event = 'firstMovement_times' #movement, feedback
    elif C.ALIGN_EVENT == 'feedback':
        event = 'feedback_times' #movement, feedback
    return event


def load_and_prepare_trials( trial_type, event_list, clean_rt, rt_variable_name):
    """Load, filter, and optionally clean the trials table.


    Steps:
    1) load from CSV
    2) drop trials with NaNs in required event columns (if provided)
    3) restrict to first 400 trials if requested
    4) add cleaned RT column and drop NaNs if requested


    Returns
    -------
    pd.DataFrame | None
    Cleaned trials table, or None if loading failed.
    """
    trials_path = os.path.join(C.DATAPATH, 'ibl_included_eids_trials_table2025_full.csv')
    try:
        trials_table = pd.read_csv(trials_path)
    except Exception as err:
        print(f'Error loading trials table: {err}')
        return None

    if event_list:
        # Keep only trials with all required event present
        trials_table['exclude_nan_event_mask'] = np.where(trials_table[event_list].notna().all(axis=1), 1, 0)
        trials_table = trials_table[trials_table['exclude_nan_event_mask'] == 1]

    if trial_type == 'first400':
        trials_table = trials_table.groupby('eid').head(400).reset_index(drop=True)

    if clean_rt:
        trials_table = clean_rt_table(trials_table, rt_variable_name)
        trials_table = trials_table[~trials_table['rt'].isna()]
    
    return trials_table


def load_sorting_and_clusters(row, one, no_iblsortor):
    """Load spikes/clusters/channels for a PID with project‑specific logic.


    For BWM (brainwidemap):
    • use revision '2024-05-06' and `good_units=True`, then merge clusters


    For Learning Lifespan:
    • try iblsort with `enforce_version=True`, fall back to `False` on
    `AssertionError`, and record the PID in `no_iblsortor` for auditing.


    Returns
    -------
    (spikes, clusters, channels, no_iblsortor)
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
    """Compute presence ratio & firing rate within the analysis window.


    The analysis window spans from min(event) - 0.4 s to max(event) + 1 s
    where `event` is chosen by `map_event()`.


    Also maps cluster atlas IDs to Beryl and merged region labels.


    Returns
    -------
    (clusters, spike_times_btw, spike_clusters)
    clusters : updated table with 'presence_ratio_poi' and 'firing_rate_poi'
    spike_times_btw : spike times within the analysis window
    spike_clusters : cluster ids aligned to `spike_times_btw`
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
    """Summarize the recording "yield" and select good clusters by region.

    Channel yield: number of distinct channels per merged Beryl region.
    Cluster yield: number of good clusters per merged Beryl region using:
        region ∈ ROIs, label == 1, FR_poi > threshold, PR_poi > threshold

    Returns
    -------
    (yield_table, clusters_ids, cluster_idx)
    yield_table : per-region counts + recording metadata
    clusters_ids : array of selected cluster IDs (good units)
    cluster_idx : boolean mask over `clusters` rows (good units)
    """
    # --- Channels: add region labels
    channels['Beryl'] = br.id2acronym(channels['atlas_id'], mapping='Beryl')
    channels['Beryl_merge'] = combine_regions(channels['Beryl'])
    # `channels` can be a dict‑like; build a DataFrame robustly
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

    # --- Clusters: select good units
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

    # Merge channel and cluster counts (outer to keep zeros)
    yield_table = pd.merge(channels_good, cluster_good, on="Beryl_merge", how="outer").reset_index(drop=True)
    yield_table['pid'] = pid
    yield_table['eid'] = eid
    yield_table['subject'] = subject
    yield_table['age_at_recording'] = age_at_recording

    return yield_table, clusters_ids, cluster_idx


def filter_spikes_by_cluster(spikes, clusters_ids, pid, pid_no_spikes):
    """Filter spikes to selected cluster ids.


    Returns
    -------
    (spike_idx, has_valid_spikes, pid_no_spikes)
    spike_idx : boolean mask over `spikes['clusters']`
    has_valid_spikes : False if no spikes survived filtering
    pid_no_spikes : updated list of PIDs with zero spikes in ROIs
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
    """Compute condition-wise and residual FF/FR time courses per unit.


    Pipeline:
    1) Compute full-trial aligned bins across all trials (reference epoch)
    2) For each signed_contrast:
    • compute aligned bins
    • FF = var(counts) / mean(counts) across trials
    • FR_mean = mean(counts / bin_size) across trials
    • Normalize bins by min-max per unit using the *full* epoch
    • FF_normed: var(normed_counts) / mean(normed_counts)
    • Accumulate (counts - condition mean) for residual FF
    3) Residual FF: var(concatenated adjusted counts) / weighted mean across conditions
    4) Full-epoch normalized FF/FR for reference


    Returns
    -------
    (df_all, df_all_conditions)
    df_all : list[pd.DataFrame]
    Per-unit residual/full summaries (use `enrich_df`).
    df_all_conditions : list[pd.DataFrame]
    Per-unit condition-split summaries (use `enrich_df_conditions`).
    """
    df_all = []
    df_all_conditions = []
    all_sc_adjusted_bins = []
    weighted_mean_sum = None
    total_weight = 0

    # --- 1) Full epoch bins across all trials (reference for normalization)
    bins_full, t_full = smoothing_sliding(
        spikes['times'][spike_idx], spikes['clusters'][spike_idx], clusters_ids,
        trials[event].values, align_epoch=event_epoch, bin_size=bin_size
    )

    # --- 2) Condition‑wise metrics
    for signed_contrast, group in trials.groupby('signed_contrast'):
        bins, t = smoothing_sliding(
            spikes['times'][spike_idx], spikes['clusters'][spike_idx], clusters_ids,
            group[event].values, align_epoch=event_epoch, bin_size=bin_size
        )
        num_trials = bins.shape[0]
        # Normalize using extrema from the full epoch (prevents condition leakage)
        sc_normalized = normalize_epoch_extremum(bins, bins_full)
        fr_normalized = normalize_epoch_extremum(bins/bin_size, bins_full/bin_size)
        # Across‑trial means for FR; keep unit×time shape
        fr_mean_across_trials = np.mean(bins/bin_size, axis=0)
        fr_normalized_mean_across_trials = np.mean(fr_normalized, axis=0)
        # FF per unit×time across trials
        with np.errstate(divide='ignore', invalid='ignore'):
            ff = np.nanvar(bins, axis=0) / np.nanmean(bins, axis=0)
        
        ff_var = np.nanvar(bins, axis=0)
        ff_mean = np.nanmean(bins, axis=0)
        ff_normed = np.nanvar(sc_normalized, axis=0) / np.nanmean(sc_normalized, axis=0)
        # Accumulate for residual FF
        sc_mean_across_trials = np.mean(bins, axis=0)
        if weighted_mean_sum is None:
            weighted_mean_sum = sc_mean_across_trials * num_trials
        else:
            weighted_mean_sum += sc_mean_across_trials * num_trials
        total_weight += num_trials

        sc_adjusted_bins = bins - sc_mean_across_trials
        all_sc_adjusted_bins.append(sc_adjusted_bins)

        # Build condition‑split rows per unit
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
    # --- 3) Residual FF pooling across conditions
    concatenated_bins = np.concatenate(all_sc_adjusted_bins, axis=0)
    weighted_mean_across_conditions = weighted_mean_sum / total_weight
    ff_residuals = np.nanvar(concatenated_bins, axis=0) / weighted_mean_across_conditions
    fr_residuals = weighted_mean_across_conditions / bin_size
    ff_var_residuals = np.nanvar(concatenated_bins, axis=0)
    ff_mean_residuals = weighted_mean_across_conditions
    ff_time = t # re‑use last t (all conditions share the same grid)

    # --- 4) Full‑epoch normalized FF/FR for reference plots
    sc_normalized_full = normalize_epoch_extremum(bins_full, bins_full)
    fr_normalized_full_mean_across_trials = np.mean(sc_normalized_full, axis=0)
    ff_full_normed = np.nanvar(sc_normalized_full, axis=0) / np.nanmean(sc_normalized_full, axis=0)
    ff_full_original = np.nanvar(bins_full, axis=0) / np.nanmean(bins_full, axis=0)
    fr_full_original = np.mean(bins_full / bin_size, axis=0)

    # Build residual/full rows per unit
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


# --- Logging setup and per‑pid processed dir ---
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

    # 1) Trials table
    trials_table = load_and_prepare_trials(C.TRIAL_TYPE, C.EVENT_LIST, C.CLEAN_RT, C.RT_VARIABLE_NAME)
    if trials_table is None:
        print("Failed to load trials.")

    # print(len(set(trials_table.eid)))

    # 2) Recording list after QC
    recordings_filtered = read_table(C.DATAPATH / 'BWM_LL_release_afterQC_df.csv')

    ba = AllenAtlas()
    br = BrainRegions()
    one = ONE()
    no_iblsortor=[]# pids for which iblsort enforce_version had to be relaxed
    pid_no_spikes=[] # pids with zero spikes in selected ROIs

    list_ind = []
    all_pids_to_use = recordings_filtered['pid'].to_list()

    # 4) Main processing loop (idempotent via .done guards)
    for index, row in recordings_filtered.iterrows():
        pid = row['pid']
        # if pid in ['57edc590-a53d-403c-9aab-d58ee51b6a24', 'daadb3f1-bef2-474e-a659-72922f3fcc5b', '61bb2bcd-37b4-4bcc-8f40-8681009a511a', 'ee2ce090-696a-40f5-8f29-7107339bf08e']:
        #     continue
        
        print(f"PID {index+1}/{len(recordings_filtered)}: {pid}")
        print(pid)

        # Idempotency guard
        done_file = processed_dir / f"{pid}.done"
        if done_file.exists():
            print(f"Skipping already processed pid: {pid}")
            continue

        print(f'processing {index + 1}/{len(recordings_filtered)}') #

        try:
            list_ind.append(index)
            eid, pname = one.pid2eid(pid)
            eid = str(eid)
            # Session‑specific trials & subject info
            trials, subject, sex, age_at_recording, sess_date = extract_mouse_info(trials_table, eid)
            # Load spikes/clusters/channels (project‑aware)
            spikes, clusters, channels, no_iblsortor = load_sorting_and_clusters(row, one, no_iblsortor)

            event = map_event()

            # Compute cluster metrics in the analysis window
            clusters, spike_times_btw, spike_clusters = compute_cluster_metrics(spikes, clusters, trials, br)

            # Select good units and summarize yield
            yield_table, clusters_ids, cluster_idx = compute_neural_yield(
                                clusters, channels, C.ROIS, C.FIRING_RATE_THRESHOLD,
                                C.PRESENCE_RATIO_THRESHOLD, pid, eid, subject, age_at_recording, br
                            )
            
            # Filter spikes to the selected units
            spike_idx, has_valid_spikes, pid_no_spikes = filter_spikes_by_cluster(spikes, clusters_ids, pid, pid_no_spikes)
            if not has_valid_spikes:
                continue

            # Compute FF/FR (condition‑wise + residual) for this pid
            df_this, df_conditions_this = compute_fano_factors(
                spikes, spike_idx, clusters_ids, trials, event, C.EVENT_EPOCH, C.BIN_SIZE,
                eid, pid, age_at_recording, sex, subject, sess_date, C.TRIAL_TYPE,
                clusters, cluster_idx
            )
            # df_all.extend(df_this)
            # df_all_conditions.extend(df_conditions_this)

            # Persist per‑pid intermediates (idempotent merges later)
            df_pid = pd.concat(df_this, ignore_index=True)
            df_cond_pid = pd.concat(df_conditions_this, ignore_index=True)

            df_pid.to_parquet(processed_dir / f"{pid}_ff.parquet", engine='pyarrow', compression='snappy')
            df_cond_pid.to_parquet(processed_dir / f"{pid}_ff_cond.parquet", engine='pyarrow', compression='snappy')
            yield_table.to_parquet(processed_dir / f"{pid}_yield.parquet", engine='pyarrow', compression='snappy')

            # Mark completion for resume‑safety
            done_file.touch()

        except Exception as err:
            # Log and continue with the next pid (robust batch behavior)
            print(f"Error on pid {pid} (index {index}): {err}")
            traceback.print_exc()

            logging.error(f"pid {pid} (index {index}) — {err}")
            logging.error(traceback.format_exc())

            continue

    # all pids expected
    expected_pids = set(recordings_filtered['pid'])

    # pids processed, with a .done file
    done_pids = {Path(f).stem for f in glob(str(processed_dir / "*.done"))}

    # any missing pid
    missing_pids = expected_pids - done_pids

    if missing_pids:
        print(f"Warning: {len(missing_pids)} pids missing results.")
        # print(f"Missing pids (sample): {list(missing_pids)[:5]}")
    else:
        print("All expected pids are present.")

    # Merge all per‑PID outputs
    df_all = pd.concat([read_table(f) for f in glob(str(processed_dir / "*_ff.parquet"))], ignore_index=True)
    df_all_conditions = pd.concat([read_table(f) for f in glob(str(processed_dir / "*_ff_cond.parquet"))], ignore_index=True)
    neural_yield_all = pd.concat([read_table(f) for f in glob(str(processed_dir / "*_yield.parquet"))], ignore_index=True)
    
    # Final save
    if save_results:
        save_results_to_parquet(
            df_all, df_all_conditions, neural_yield_all, suffix="2025")

# %%
