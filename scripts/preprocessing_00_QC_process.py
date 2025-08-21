"""
generate a QC summary table for the datasets.
output: QC_table_wide.csv

BWM_LL_release_afterQC_df.csv

figures/QC_funnel.pdf
"""

# %% Imports
import config as C
from pathlib import Path
import os
import pandas as pd
from one.api import ONE
from iblatlas.regions import BrainRegions

from scripts.utils.query import bwm_query, lifespan_query
from scripts.utils.neuron_utils import combine_regions
from scripts.utils.behavior_utils import filter_trials
from scripts.utils.io import read_table


# % Helper: Summary generator
def summarize_QC(name, level, df, clus_df):
    return {
        'name': name,
        'level': level,
        'n_sessions': df['eid'].nunique(),
        'n_probes': df['pid'].nunique(),
        'n_units': clus_df.loc[clus_df['eid'].isin(df['eid'].unique())]['uuids'].nunique(),
    }

# % Core QC Function
def filter_sessions_insertions(query_insertions, trials_df, clus_df, rt_range, min_errors, min_qc, min_sessions):
    outcomes = []
    
    valid_sessions = query_insertions.copy()

    # --- Level 1: Initial state
    outcomes.append(summarize_QC('Session and insertion QC', 'probe,session', valid_sessions, clus_df))

    # --- Level 2: Filter by RT & event
    trials_df["response_times_from_stim"] = trials_df["response_times"] - trials_df["stimOn_times"]
    trials_df = filter_trials(trials_df, exclude_nan_event_trials=True, trial_type='all',
                              event_list=C.event_list, clean_rt=True,
                              rt_variable=C.rt_variable_name, rt_cutoff=rt_range)
    valid_sessions = valid_sessions[valid_sessions['eid'].isin(trials_df['eid'].unique())]
    outcomes.append(summarize_QC('Reaction time and missing events', 'trials', valid_sessions, clus_df))

    # --- Level 3: Sessions with at least X error trials
    trials_agg = trials_df.groupby('eid').agg(
        n_trials=('eid', 'count'),
        n_error=('feedbackType', lambda x: (x == -1).sum())
    )
    valid_error_sessions = trials_agg[trials_agg['n_error'] >= min_errors].index
    valid_sessions = valid_sessions[valid_sessions['eid'].isin(valid_error_sessions)]
    outcomes.append(summarize_QC('Minimum 3 error trials', 'session', valid_sessions, clus_df))

    # --- Level 4: Neuron-level QC
    clus_df = clus_df[clus_df['eid'].isin(valid_sessions['eid'].unique())]
    clus_df = clus_df[(clus_df['label'] >= min_qc) & (clus_df['firing_rate'] > 1) & (clus_df['presence_ratio'] > 0.95)]
    outcomes.append(summarize_QC('Single unit QC', 'neuron', clus_df, clus_df))

    # --- Level 5: Restrict to specific ROIS
    br = BrainRegions()
    clus_df['Beryl'] = br.id2acronym(clus_df['atlas_id'], mapping='Beryl')
    clus_df['Beryl_merge'] = combine_regions(clus_df['Beryl'])
    clus_df = clus_df[clus_df['Beryl_merge'].isin(C.ROIS)]
    outcomes.append(summarize_QC('ROIS only', 'neuron', clus_df, clus_df))

    # --- Level 6: Keep only regions with at least X sessions
    units_count = clus_df.groupby(['Beryl_merge', 'eid'])['cluster_id'].count().reset_index()
    sessions_per_region = units_count.groupby('Beryl_merge')['eid'].nunique()
    valid_regions = sessions_per_region[sessions_per_region >= min_sessions].index
    region_sessions = units_count[units_count['Beryl_merge'].isin(valid_regions)]

    clus_df = pd.merge(clus_df, region_sessions[['eid', 'Beryl_merge']], on=['eid', 'Beryl_merge'], how='inner')
    outcomes.append(summarize_QC('Minimum 2 sessions per region', 'neuron', clus_df, clus_df))

    return outcomes, clus_df['pid'].unique(), clus_df['eid'].unique()


def main():
    one = ONE()
    datapath_bwm = C.DATAPATH / 'bwm_tables'
    datapath_lifespan = C.DATAPATH / 'lifespan_tables'

    # Query metadata
    bwm_df = bwm_query(alignment_resolved=True, return_details=False)
    lifespan_df = lifespan_query(one=one, alignment_resolved=True, return_details=False)
    bwm_df['project'] = 'brainwidemap'
    lifespan_df['project'] = 'learninglifespan'
    full_release_df = pd.concat([bwm_df, lifespan_df])

    # Load tables
    try:
        clus_df_bwm = read_table(datapath_bwm / 'clusters.pqt')
        trials_df_bwm = read_table(datapath_bwm / 'trials.pqt')
    except Exception as e:
        raise RuntimeError("Missing BWM cluster/trial data. Please download first.") from e

    clus_df_lifespan = read_table(datapath_lifespan / 'clusters.pqt')
    trials_df_lifespan = read_table(datapath_lifespan / 'trials.pqt')

    # Apply QC pipeline
    outcomes_bwm, pids_bwm, _ = filter_sessions_insertions(
        bwm_df, trials_df_bwm, clus_df_bwm, rt_range=C.RT_CUTOFF, min_errors=3, min_qc=1.0, min_sessions=2
    )
    outcomes_lifespan, pids_lifespan, _ = filter_sessions_insertions(
        lifespan_df, trials_df_lifespan, clus_df_lifespan, rt_range=C.RT_CUTOFF, min_errors=3, min_qc=1.0, min_sessions=2
    )

    # Merge and save QC summary
    df_bwm = pd.DataFrame(outcomes_bwm)
    df_lifespan = pd.DataFrame(outcomes_lifespan)

    # rename columns for clarity
    df_bwm = df_bwm.rename(columns={
        'n_sessions': 'n_sessions_IBL',
        'n_probes': 'n_probes_IBL'
    })[['name', 'level', 'n_sessions_IBL', 'n_probes_IBL']]

    df_lifespan = df_lifespan.rename(columns={
        'n_sessions': 'n_sessions_New',
        'n_probes': 'n_probes_New'
    })[['name', 'level', 'n_sessions_New', 'n_probes_New']]

    order = df_bwm['name'].tolist()

    # merge the two DataFrames
    qc_table_wide = pd.merge(df_bwm, df_lifespan, on=['name', 'level'], how='outer')
    qc_table_wide['name'] = pd.Categorical(qc_table_wide['name'], categories=order, ordered=True)
    qc_table_wide = qc_table_wide.sort_values('name').reset_index(drop=True)
    qc_table_wide = qc_table_wide.rename(columns={'name': 'QC item'})

    qc_table_wide.to_csv(C.DATAPATH / 'test0816QC_table_wide.csv', index=False)

    # Save remaining sessions
    final_pids = list(set(pids_bwm).union(set(pids_lifespan)))
    filtered_df = full_release_df[full_release_df['pid'].isin(final_pids)]
    print(filtered_df.pid.nunique()) #507 insertions
    filtered_df.to_csv(C.DATAPATH / 'test0816BWM_LL_release_afterQC_df.csv', index=False)


if __name__ == "__main__":
    main()

# %%
