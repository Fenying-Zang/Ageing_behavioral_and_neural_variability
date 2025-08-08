
#%%
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from utils.config import datapath, align_event, trial_type, PRE_TIME, POST_TIME, tolerance


def load_timecourse_data(df_path):
    """
    Load df_all_conditions containing FF and FR timecourses.
    """
    return pd.read_parquet(df_path)

def extract_pre_post_values(df, pre_time=PRE_TIME, post_time=POST_TIME, tol=tolerance):
    """
    Extract pre/post values at specified timepoints for FF and FR.
    Assumes df has one row per neuron × contrast × timepoint.
    """
    metrics = []
    grouped = df.groupby(['uuids', 'signed_contrast'])
    for (uid, contrast), group in grouped:
        # n_trials = len(group)
        # print(uid, contrast)
        # print(n_trials)
        # print(len(group['']))
        pre_row = group[np.abs(group['timepoints'] - pre_time) < tol]
        post_row = group[np.abs(group['timepoints'] - post_time) < tol]

        if len(pre_row) == 1 and len(post_row) == 1:
            ff_delta = post_row['FFs'].values[0] - pre_row['FFs'].values[0]
            fr_delta = post_row['frs'].values[0] - pre_row['frs'].values[0]
            metrics.append({
                'uuids': uid,
                'signed_contrast': contrast,
                'n_trials': pre_row['n_trials'].values[0],
                'abs_contrast': abs(contrast)/100,   # 新增
                'pre_ff': pre_row['FFs'].values[0],
                'post_ff': post_row['FFs'].values[0],
                'ff_quench': ff_delta,
                'pre_fr': pre_row['frs'].values[0],
                'post_fr': post_row['frs'].values[0],
                'fr_delta': fr_delta,
                # carry metadata
                'cluster_region': pre_row['cluster_region'].values[0],
                'mouse_age': pre_row['mouse_age'].values[0],
                'mouse_sub_name': pre_row['mouse_sub_name'].values[0],
                'session_pid': pre_row['session_pid'].values[0],
                'session_eid': pre_row['session_eid'].values[0]
                #TODO: if needed, add more 
            })
    return pd.DataFrame(metrics)

def extract_pre_post_from_df_meansub(df, pre_time=PRE_TIME, post_time=POST_TIME, tol=tolerance):
    """
    Extract pre/post FF and FR from df_all (mean-subtracted version), no contrast splitting.
    """
    metrics = []
    grouped = df.groupby('uuids')
    for uid, group in grouped:
        pre_row = group[np.abs(group['timepoints'] - pre_time) < tol]
        post_row = group[np.abs(group['timepoints'] - post_time) < tol]
        if len(pre_row) == 1 and len(post_row) == 1:
            ff_delta = post_row['FFs_residuals'].values[0] - pre_row['FFs_residuals'].values[0]
            fr_delta = post_row['frs_residuals'].values[0] - pre_row['frs_residuals'].values[0]
            metrics.append({
                'uuids': uid,
                'n_trials': pre_row['n_trials'].values[0],
                'pre_ff': pre_row['FFs_residuals'].values[0],
                'post_ff': post_row['FFs_residuals'].values[0],
                'ff_quench': ff_delta,
                'pre_fr': pre_row['frs_residuals'].values[0],
                'post_fr': post_row['frs_residuals'].values[0],
                'fr_delta': fr_delta,
                'cluster_region': pre_row['cluster_region'].values[0],
                'mouse_age': pre_row['mouse_age'].values[0],
                'mouse_sub_name': pre_row['mouse_sub_name'].values[0],
                'session_pid': pre_row['session_pid'].values[0],
                'session_eid': pre_row['session_eid'].values[0]
           })
    return pd.DataFrame(metrics)


def compute_modulation_index(df, metric_col):
    """
    Compute slope of metric_col vs. signed_contrast as contrast modulation index.
    Returns: DataFrame with one row per neuron (uuids).
    """
    slopes = []
    grouped = df.groupby('uuids')
    for uid, group in grouped:
        # if group['abs_contrast'].nunique() < 5:
        #     continue  # skip if contrast not enough
        group = group.dropna(subset=['abs_contrast', metric_col])
        if len(group) < 5:
            print(f"Skipping {uid}: too few valid contrast levels ({len(group)})")
            continue

        # X = group[['signed_contrast']].values.reshape(-1, 1)
        X = group[['abs_contrast']].values.reshape(-1, 1)

        y = group[metric_col].values
        model = LinearRegression().fit(X, y)
        slopes.append({
            'uuids': uid,
            f'{metric_col}_modulation': model.coef_[0],
            'cluster_region': group['cluster_region'].iloc[0],
            'mouse_age': group['mouse_age'].iloc[0],
            'mouse_sub_name': group['mouse_sub_name'].iloc[0],
            'session_pid': group['session_pid'].iloc[0],
            'session_eid': group['session_eid'].iloc[0],
        })
    return pd.DataFrame(slopes)


if __name__ == "__main__":

    df_cond_path = datapath / f"ibl_BWMLL_FFs_{align_event}_{trial_type}_conditions_2025_merged.parquet"
    df_meansub_path = datapath / f"ibl_BWMLL_FFs_{align_event}_{trial_type}_2025_merged.parquet"
    out_path_cond = datapath / "neural_metrics_summary_conditions_merged.parquet"
    out_path_meansub = datapath / "neural_metrics_summary_meansub_merged.parquet"


    # print("Loading df_all_conditions...")
    # df_cond = load_timecourse_data(df_cond_path)
    # print("Extracting condition-based metrics...")
    # df_summary = extract_pre_post_values(df_cond)

    # print("Computing contrast modulation indices...")
    # ff_mod = compute_modulation_index(df_summary, 'ff_quench')
    # fr_mod = compute_modulation_index(df_summary, 'fr_delta')

    # print("Merging condition-based results...")
    # final_cond = df_summary.merge(ff_mod, on=['uuids', 'cluster_region', 'mouse_age', 'session_pid'], how='left')
    # final_cond = final_cond.merge(fr_mod, on=['uuids', 'cluster_region', 'mouse_age', 'session_pid'], how='left')

    # print(f"Saving to {out_path_cond}")
    # final_cond.to_parquet(out_path_cond, index=False)

    print("\nLoading df_all (mean-subtracted FF)...")
    df_meansub = load_timecourse_data(df_meansub_path)
    print("Extracting mean-subtracted pre/post metrics...")
    final_meansub = extract_pre_post_from_df_meansub(df_meansub)

    print(f"Saving to {out_path_meansub}")
    final_meansub.to_parquet(out_path_meansub, index=False)


