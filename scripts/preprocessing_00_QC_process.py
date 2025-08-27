"""
Quality Control on sessions, probes, trials, and neurons
Outputs:
    - QC_summary_table_wide.csv  (QC summary table for the datasets) -->Table 2 in the manuscript
    - BWM_LL_release_afterQC_df.csv (filtered pids for downstream analyses)

"""
# %% Imports
import config as C
import pandas as pd
from one.api import ONE
from iblatlas.regions import BrainRegions

from scripts.utils.query import bwm_query, lifespan_query
from scripts.utils.neuron_utils import combine_regions
from scripts.utils.behavior_utils import filter_trials
from scripts.utils.io import read_table, setup_logging
import logging
log = logging.getLogger(__name__)

# % Helper: Summary generator
def summarize_QC(name, level, df, clus_df):
    """
    Build a compact QC summary at a given level.

    Parameters
    ----------
    name : str
        A readable step name (e.g., "Single unit QC").
    level : str
        Granularity label (e.g., "trials", "neuron").
    df : pd.DataFrame
        The current 'session/insertion' frame to count unique eids/pids from.
    clus_df : pd.DataFrame
        cluster (neuron, or unit) table; used to count unique clusters included in current QC step.
    """
    # Limit cluster table to sessions present in df before counting units
    _eids = df['eid'].unique()
    n_units = clus_df.loc[clus_df['eid'].isin(_eids), 'uuids'].nunique()

    return {
        'name': name,
        'level': level,
        'n_sessions': df['eid'].nunique(),
        'n_probes': df['pid'].nunique(),
        'n_units': n_units,
    }


# % Core QC Function
def filter_sessions_insertions(query_insertions, trials_df, clus_df, rt_range, min_errors, min_qc, min_sessions):
    """
    Apply the QC steps from session/probe, trials to neuron level.

    Steps:
      1) Initial state
      2) Filter trials by RT & event completeness
      3) Keep sessions with >= min_errors error trials
      4) Neuron-level QC (label >= min_qc, FR > 1 sp/s, presence_ratio > 0.95)
      5) Keep only specific ROIs (C.ROIS)
      6) Keep only regions with >= min_sessions sessions

    Returns
    -------
    outcomes : list[dict]
        QC summaries at each step.
    valid_pids : np.ndarray
        Remaining pid list after QC.
    valid_eids : np.ndarray
        Remaining eid list after QC.
    """
    outcomes = []

    # Make explicit copies to avoid chained-assignment issues
    valid_sessions = query_insertions.copy()
    trials_df = trials_df.copy()
    clus_df = clus_df.copy()

    # --- Level 1: Initial state
    outcomes.append(summarize_QC('Session and insertion QC', 'probe,session', valid_sessions, clus_df))

    # --- Level 2: Filter by RT & event 
    trials_df = trials_df.loc[trials_df['eid'].isin(valid_sessions['eid'].unique())].copy()
    print(f"[QC] #Sessions before RT/event filtering: {trials_df['eid'].nunique()}")

    # Compute response times from stimulus onset and apply trial filter
    trials_df["response_times_from_stim"] = trials_df["response_times"] - trials_df["stimOn_times"]
    trials_df = filter_trials(
        trials_df,
        exclude_nan_event_trials=True,
        trial_type='all',
        event_list=C.EVENT_LIST,
        clean_rt=True,
        rt_variable='response_times_from_stim',
        rt_cutoff=rt_range
    )
    valid_sessions = valid_sessions[valid_sessions['eid'].isin(trials_df['eid'].unique())]
    outcomes.append(summarize_QC('Response time and missing events', 'trials', valid_sessions, clus_df))

    # --- Level 3: Sessions with at least X error trials
    trials_agg = trials_df.groupby('eid').agg(
        n_trials=('eid', 'count'),
        n_error=('feedbackType', lambda x: (x == -1).sum())
    )
    valid_error_sessions = trials_agg.index[trials_agg['n_error'] >= min_errors]
    valid_sessions = valid_sessions[valid_sessions['eid'].isin(valid_error_sessions)]
    outcomes.append(summarize_QC(f'Minimum {min_errors} error trials', 'session', valid_sessions, clus_df))

    # --- Level 4: Neuron-level QC
    # clus_df = clus_df[clus_df['eid'].isin(valid_sessions['eid'].unique())].copy()
    clus_df = clus_df[clus_df['pid'].isin(valid_sessions['pid'].unique())].copy()

    clus_df = clus_df[
        (clus_df['label'] >= min_qc) &
        (clus_df['firing_rate'] > 1) &
        (clus_df['presence_ratio'] > 0.95)
    ].copy()
    outcomes.append(summarize_QC('Single unit QC', 'neuron', clus_df, clus_df))

    # --- Level 5: Restrict to specific ROIs
    br = BrainRegions()

    clus_df['Beryl'] = br.id2acronym(clus_df['atlas_id'], mapping='Beryl')
    clus_df['Beryl_merge'] = combine_regions(clus_df['Beryl'])
    clus_df = clus_df[clus_df['Beryl_merge'].isin(C.ROIS)].copy()
    outcomes.append(summarize_QC('ROIs only', 'neuron', clus_df, clus_df))

    # --- Level 6: Keep only regions with at least X sessions
    units_count = clus_df.groupby(['Beryl_merge', 'eid'])['cluster_id'].count().reset_index()
    sessions_per_region = units_count.groupby('Beryl_merge')['eid'].nunique()
    valid_regions = sessions_per_region.index[sessions_per_region >= min_sessions]
    region_sessions = units_count[units_count['Beryl_merge'].isin(valid_regions)]

    # Inner-join keeps only rows from clus_df that belong to regions with enough sessions
    clus_df = pd.merge(
        clus_df,
        region_sessions[['eid', 'Beryl_merge']],
        on=['eid', 'Beryl_merge'],
        how='inner'
    )
    outcomes.append(summarize_QC(f'Minimum {min_sessions} sessions per region', 'neuron', clus_df, clus_df))

    return outcomes, clus_df['pid'].unique(), clus_df['eid'].unique()


def main():
    from scripts.utils.io import setup_logging
    setup_logging()

    one = ONE()

    # Query metadata from release table (BWM) and database (Lifespan) (session/probe tables)
    bwm_df = bwm_query(alignment_resolved=True, return_details=False)
    lifespan_df = lifespan_query(one=one, alignment_resolved=True, return_details=False)
    
    # Exclude pids without ilblsortor early so all downstream tables stay consistent
    lifespan_df = lifespan_df[~lifespan_df['pid'].isin(C.PIDS_WITHOUT_ILBLSORTOR)].copy()

    bwm_df['project'] = 'brainwidemap'
    lifespan_df['project'] = 'learninglifespan'

    full_release_df = pd.concat([bwm_df, lifespan_df], ignore_index=True)

    # Load precomputed tables
    try:
        clus_df_bwm = read_table(C.DATAPATH / 'bwm_tables' / 'clusters.pqt')
        trials_df_bwm = read_table(C.DATAPATH / 'bwm_tables' / 'trials.pqt')
    except Exception as e:
        raise RuntimeError("Missing BWM cluster/trial data. Please download first.") from e

    clus_df_lifespan = read_table(C.DATAPATH / 'lifespan_tables' / 'clusters.pqt')
    trials_df_lifespan = read_table(C.DATAPATH / 'lifespan_tables' / 'trials.pqt')

    # Apply QC pipeline separately to BWM and Lifespan
    outcomes_bwm, pids_bwm, _ = filter_sessions_insertions(
        bwm_df, trials_df_bwm, clus_df_bwm,
        rt_range=C.RT_CUTOFF, min_errors=3, min_qc=1.0, min_sessions=2
    )
    outcomes_lifespan, pids_lifespan, _ = filter_sessions_insertions(
        lifespan_df, trials_df_lifespan, clus_df_lifespan,
        rt_range=C.RT_CUTOFF, min_errors=3, min_qc=1.0, min_sessions=2
    )

    # Build QC table (wide format for side-by-side comparison)
    df_bwm = pd.DataFrame(outcomes_bwm).rename(columns={
        'n_sessions': 'n_sessions_IBL',
        'n_probes': 'n_probes_IBL'
    })[['name', 'level', 'n_sessions_IBL', 'n_probes_IBL']]

    df_lifespan = pd.DataFrame(outcomes_lifespan).rename(columns={
        'n_sessions': 'n_sessions_New',
        'n_probes': 'n_probes_New'
    })[['name', 'level', 'n_sessions_New', 'n_probes_New']]

    # Keep the same step order as BWM
    order = df_bwm['name'].tolist()
    qc_table_wide = pd.merge(df_bwm, df_lifespan, on=['name', 'level'], how='outer')
    qc_table_wide['name'] = pd.Categorical(qc_table_wide['name'], categories=order, ordered=True)
    qc_table_wide = qc_table_wide.sort_values('name').reset_index(drop=True)
    qc_table_wide = qc_table_wide.rename(columns={'name': 'QC item'})

    outfile_qc_summary = C.RESULTSPATH / 'QC_summary_table_wide.csv'
    qc_table_wide.to_csv(outfile_qc_summary, index=False)
    log.info(f"[Saved table] {outfile_qc_summary.resolve()}")

    # Save remaining pids (union of both datasets after their respective QC)
    final_pids = sorted(set(pids_bwm).union(set(pids_lifespan)))

    # Keep only rows from the globally pre-filtered release set
    filtered_df = full_release_df[full_release_df['pid'].isin(final_pids)].copy()
    print(f"[QC] #Insertions after full QC: {filtered_df.pid.nunique()}")

    outfile_filtered_recordings = C.DATAPATH / 'BWM_LL_release_afterQC_df.csv'
    filtered_df.to_csv(outfile_filtered_recordings, index=False)
    log.info(f"[Saved table] {outfile_filtered_recordings.resolve()}")

if __name__ == "__main__":
    main()

# %%
