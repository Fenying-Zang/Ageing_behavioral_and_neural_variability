"""
behavior_07_process_movement_time_courses.py

Compute per-trial wheel & DLC speeds/accelerations aligned to stim,
aggregate per session & signed_contrast into 120-sample timecourses,
explode to long format, pre-filter by video/DLC QC, run session-level QC,
and save raw + QC-filtered outputs.

Outputs
-------
C.DATAPATH / "ibl_wheel&dlc_movement_timecourse_2025.parquet"
C.DATAPATH / "ibl_QCfiltered_wheel_cam_results.parquet"
C.DATAPATH / "ibl_QC_summary_movement.csv"
"""
#%%
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

from one.api import ONE
from brainbox.io.one import SessionLoader
from one.alf import spec  # for QC enum ordering

# your helpers
from scripts.utils.behavior_utils import filter_trials
from scripts.utils.dlc_revised import likelihood_threshold, get_speed
import config as C
from scripts.utils.plot_utils import figure_style

# =====================
# Tunables
# =====================
T_BIN = 0.02
WINDOW_LEN = 2.0
WINDOW_LAG = -0.5
N_SAMPLES = 120
SAMPLE_RATE_WHEEL = 1000  # Hz

SAMPLING = {'left': 60, 'right': 150, 'body': 30}
CAM_LIST = ['left', 'body']
FEATURES_SIDE = ['paw_r', 'paw_l', 'nose_tip']
FEATURES_BODY = ['tail_start']  # leave empty if unused

TRIALS_CSV = "ibl_included_eids_trials_table2025_full.csv"
SESSIONS_CSV = "BWM_LL_release_afterQC_df.csv"

RAW_OUT = "ibl_wheel&dlc_movement_timecourse_2025.parquet"
QC_OUT = "ibl_QCfiltered_wheel_cam_results.parquet"
QC_SUMMARY_CSV = "ibl_QC_summary_movement.csv"

EVENT_COL = 'stimOn_times'

# QC thresholds / window
MIN_VALID_RATIO = 0.60                # require >=60% finite rows for each metric group present
CLIP_WINDOW = (-0.2, 0.8)             # limit for plotting/QC
VIDEO_QC_MIN = 'FAIL'                 # keep NOT_SET/PASS/WARNING/FAIL (exclude CRITICAL)
DLC_QC_MIN = 'FAIL'


# =====================
# Utilities
# =====================
def plt_window(ev_times):
    start = ev_times + WINDOW_LAG
    end = ev_times + WINDOW_LAG + WINDOW_LEN

    return np.asarray(start), np.asarray(end)

def insert_idx(time_array, values):
    arr = np.asarray(time_array)
    vals = np.asarray(values)
    idx = np.searchsorted(arr, vals, side='left')
    idx[idx == len(arr)] -= 1
    left = np.clip(idx - 1, 0, len(arr) - 1)
    pick_left = np.abs(vals - arr[left]) < np.abs(vals - arr[idx])
    idx[pick_left] = left[pick_left]
    idx[idx < 0] = 0
    return idx.astype(int)

def resample_1d(x, new_len=N_SAMPLES):
    x = np.asarray(x, dtype=float)
    if len(x) == new_len:
        return x
    x_old = np.linspace(0.0, 1.0, len(x))
    x_new = np.linspace(0.0, 1.0, new_len)
    f = interp1d(x_old, x, kind='linear', bounds_error=False, fill_value="extrapolate")
    return f(x_new)

def make_time_vector(length, sampling_hz, start=WINDOW_LAG):
    return np.arange(length) / float(sampling_hz) + start


# =====================
# Video/DLC QC helpers
# =====================
def filter_video_data(one, eids, camera='left', min_video_qc='FAIL', min_dlc_qc='FAIL'):
    """
    Filter sessions whose video & DLC QC meet a minimum threshold for one camera.
    QC ordering (lower is better): NOT_SET < PASS < WARNING < FAIL < CRITICAL
    """
    if isinstance(eids, str):
        eids = [eids]
    assert isinstance(eids, list), 'eids must be a list of session UUIDs'

    vid_key = f'video{camera.capitalize()}'
    dlc_key = f'dlc{camera.capitalize()}'

    passing_vid, passing_dlc = set(), set()

    for eid in eids:
        det = one.get_details(eid, full=True) or {}
        eqc = det.get('extended_qc', {}) or {}

        # VIDEO
        if min_video_qc is None:
            passing_vid.add(eid)
        else:
            if vid_key in eqc:
                ok = spec.QC[eqc[vid_key]].value <= spec.QC[min_video_qc].value
                if ok:
                    passing_vid.add(eid)
            else:
                # missing -> treat as NOT_SET
                if spec.QC['NOT_SET'].value <= spec.QC[min_video_qc].value:
                    passing_vid.add(eid)

        # DLC
        if min_dlc_qc is None:
            passing_dlc.add(eid)
        else:
            if dlc_key in eqc:
                ok = spec.QC[eqc[dlc_key]].value <= spec.QC[min_dlc_qc].value
                if ok:
                    passing_dlc.add(eid)
            else:
                if spec.QC['NOT_SET'].value <= spec.QC[min_dlc_qc].value:
                    passing_dlc.add(eid)

    return sorted(passing_vid.intersection(passing_dlc))


def filter_video_data_multi(one, eids, cameras=('left', 'body'),
                            min_video_qc='FAIL', min_dlc_qc='FAIL', require='all'):
    """
    Apply QC for multiple cameras and combine.
    require : {'all','any'}
      - 'all'  → keep sessions that pass for every listed camera
      - 'any'  → keep sessions that pass for at least one listed camera
    """
    cams = list(cameras)
    sets = []
    for cam in cams:
        ok = set(filter_video_data(one, eids, camera=cam,
                                   min_video_qc=min_video_qc, min_dlc_qc=min_dlc_qc))
        sets.append(ok)
    keep = set.intersection(*sets) if require == 'all' else set.union(*sets)
    return sorted(keep)


# =====================
# Loaders
# =====================
def load_sessions():
    df = pd.read_csv(C.DATAPATH / SESSIONS_CSV)
    return df[['eid', 'date', 'subject', 'lab', 'project']].drop_duplicates().reset_index(drop=True)


def load_trials():
    trials = pd.read_csv(C.DATAPATH / TRIALS_CSV)
    event_list = [
        'stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
        'firstMovement_times', 'response_times', 'feedbackType'
    ]
    trials = filter_trials(
        trials, exclude_nan_event_trials=True, trial_type=C.TRIAL_TYPE,
        event_list=event_list, clean_rt=True,
        rt_variable='response_times_from_stim', rt_cutoff=[0.08, 2]
    )
    trials['age_group'] = (trials['mouse_age'] > 300).map({True: "old", False: "young"})
    return trials


# =====================
# Core computations
# =====================
def compute_dlc_feature_slices(one: ONE, eid, trials_df):
    """
    For each camera & feature, compute per-trial speed and abs(acceleration) slices,
    resampled to N_SAMPLES. Returns dict: key -> list[trial arrays], plus single 'time_dlc'.
    """
    out = {}
    for cam in CAM_LIST:
        features = FEATURES_BODY if cam == 'body' else FEATURES_SIDE
        try:
            cam_times = one.load_dataset(eid, f'alf/_ibl_{cam}Camera.times.npy')
            sess_loader = SessionLoader(one=one, eid=eid)
            sess_loader.load_pose(likelihood_thr=0.9, views=[cam])
            dlc_df_raw = sess_loader.pose[f'{cam}Camera']
        except Exception:
            continue

        # drop 'times' column added by SessionLoader; apply likelihood threshold
        dlc_df = dlc_df_raw.drop(columns=['times'], errors='ignore')
        dlc_df = likelihood_threshold(dlc_df)

        # hard-trim to identical length to silence inconsistencies
        cam_times = np.asarray(cam_times).astype(float).squeeze()
        n = min(len(cam_times), len(dlc_df))
        cam_times = cam_times[-n:]
        dlc_df = dlc_df.iloc[-n:, :].reset_index(drop=True)

        # trial windows in camera time
        start_window, end_window = plt_window(trials_df[EVENT_COL].to_numpy())
        try:
            start_idx = insert_idx(cam_times, start_window)
            end_idx = (start_idx + (WINDOW_LEN * SAMPLING[cam])).astype(int)
        except Exception:
            continue

        for feature in features:
            try:
                speeds = get_speed(dlc_df, cam_times, camera=cam, feature=feature)
            except Exception:
                continue

            # abs acceleration mapped onto cam_times
            accel = np.abs(np.diff(speeds)) * SAMPLING[cam]
            accel_t = cam_times[:-1] + np.diff(cam_times) / 2
            try:
                accel_map = interp1d(accel_t, accel, bounds_error=False, fill_value="extrapolate")(cam_times)
            except Exception:
                accel_map = np.pad(accel, (1, 0), mode='edge')

            sp_slices, ac_slices = [], []
            for i in range(len(start_idx)):
                sl = slice(start_idx[i], min(end_idx[i], len(cam_times)))
                sp_slices.append(resample_1d(speeds[sl], N_SAMPLES))
                ac_slices.append(resample_1d(accel_map[sl], N_SAMPLES))

            out[f'speed_{feature}'] = sp_slices
            out[f'speed_{feature}_acceleration'] = ac_slices

        # set canonical DLC time vector once (prefer first cam in CAM_LIST, i.e., 'left')
        if 'time_dlc' not in out:
            out['time_dlc'] = make_time_vector(N_SAMPLES, SAMPLING[cam], WINDOW_LAG)

    return out


def compute_wheel_slices(one: ONE, eid, trials_df):
    """
    Per-trial wheel velocity & acceleration (abs), resampled to N_SAMPLES.
    """
    try:
        sess_loader = SessionLoader(one=one, eid=eid)
        sess_loader.load_wheel()
        wheel = sess_loader.wheel
    except Exception:
        return {}

    w_times = wheel['times'].to_numpy()
    vel = np.abs(wheel['velocity'].to_numpy())
    acc = np.abs(wheel['acceleration'].to_numpy())

    start_window, _ = plt_window(trials_df[EVENT_COL].to_numpy())
    try:
        start_idx = insert_idx(w_times, start_window)
        end_idx = (start_idx + int(WINDOW_LEN * SAMPLE_RATE_WHEEL)).astype(int)
    except Exception:
        return {}

    sp_slices, ac_slices = [], []
    for i in range(len(start_idx)):
        sl = slice(start_idx[i], min(end_idx[i], len(w_times)))
        sp_slices.append(resample_1d(vel[sl], N_SAMPLES))
        ac_slices.append(resample_1d(acc[sl], N_SAMPLES))

    out = {
        'wheel_velocity': sp_slices,
        'wheel_acceleration': ac_slices,
        # display on same x-axis scale as DLC for plotting
        'time_wheel': make_time_vector(N_SAMPLES, 60, WINDOW_LAG)
    }
    return out


def aggregate_session_by_contrast(trials_df, per_trial):
    """
    For each signed_contrast, compute per-timebin mean (ave_) and std (vbt_) across trials,
    plus cv_ = vbt_/ave_. Returns list of per-contrast DataFrames for this session.
    """
    out = []
    items = [k for k in per_trial.keys() if k not in ('time_dlc', 'time_wheel')]
    # prefer DLC time; else wheel; else a default linspace
    time_vec = per_trial.get('time_dlc', per_trial.get('time_wheel',
                    np.linspace(WINDOW_LAG, WINDOW_LAG + WINDOW_LEN, N_SAMPLES)))

    # trials_df has 0..n-1 indices (reset_index applied upstream), aligned with per-trial list order
    for scontrast, grp in trials_df.groupby('signed_contrast'):
        vbt_dict = {'time_dlc': time_vec, 'signed_contrast': scontrast}
        idxs = grp.index.values  # positions within trials_df

        for item in items:
            slices = per_trial.get(item, [])
            if not len(slices):
                continue
            try:
                stack = np.stack([slices[i] for i in idxs])
            except Exception:
                stack = np.stack(slices[:len(idxs)])

            ave = np.nanmean(stack, axis=0)
            vbt = np.nanstd(stack, axis=0)
            eps = 1e-12
            cv = vbt / (ave + eps)

            vbt_dict[f'ave_{item}'] = ave
            vbt_dict[f'vbt_{item}'] = vbt
            vbt_dict[f'cv_{item}'] = cv

        # session metadata
        vbt_dict['eid'] = grp['eid'].iloc[0]
        vbt_dict['age_at_recording'] = grp['mouse_age'].iloc[0]
        vbt_dict['mouse_name'] = grp['mouse_name'].iloc[0]
        vbt_dict['mouse_sex'] = grp['mouse_sex'].iloc[0]

        out.append(pd.DataFrame(vbt_dict))
    return out


def process_one_session(one: ONE, eid, trials_all):
    trials_df = trials_all[trials_all['eid'] == eid].reset_index(drop=True)
    if trials_df.empty:
        return []
    per_trial = {}
    per_trial.update(compute_dlc_feature_slices(one, eid, trials_df))
    per_trial.update(compute_wheel_slices(one, eid, trials_df))
    if not per_trial:
        return []
    return aggregate_session_by_contrast(trials_df, per_trial)


# =====================
# Save (raw long)
# =====================
def explode_to_long(all_session_dfs):
    if not all_session_dfs:
        return pd.DataFrame()
    wide = pd.concat(all_session_dfs, ignore_index=True)
    explode_cols = ['time_dlc'] + [c for c in wide.columns if c.startswith(('ave_', 'vbt_', 'cv_'))]
    long = wide.explode(explode_cols, ignore_index=True)
    for c in explode_cols:
        long[c] = pd.to_numeric(long[c], errors='coerce')
    # keep plotting/QC window
    tol = float(C.TOLERANCE)
    long = long[(long['time_dlc'] >= CLIP_WINDOW[0] - tol) & (long['time_dlc'] <= CLIP_WINDOW[1] + tol)]
    return long

def save_parquet(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[OK] saved: {out_path}  (rows={len(df):,})")


# =====================
# Session-level QC
# =====================
WHEEL_KEYS = ["ave_wheel_velocity", "cv_wheel_velocity"]
DLC_KEYS = [
    "ave_speed_paw_l", "ave_speed_paw_r", "ave_speed_nose_tip",
    "cv_speed_paw_l",  "cv_speed_paw_r",  "cv_speed_nose_tip"
]

def valid_ratio(group, cols):
    present = [c for c in cols if c in group.columns]
    if not present:
        return np.nan
    m = np.ones(len(group), dtype=bool)
    for c in present:
        vals = pd.to_numeric(group[c], errors='coerce').values
        m &= np.isfinite(vals)
    return float(m.sum() / len(group)) if len(group) else 0.0

def qc_sessions(long_df, min_valid_ratio=MIN_VALID_RATIO):
    if long_df.empty:
        return long_df.copy(), pd.DataFrame(columns=['eid', 'wheel_ratio', 'dlc_ratio', 'keep', 'n_rows'])
    if 'eid' not in long_df.columns:
        raise KeyError("Expected 'eid' column.")
    rows, keep_eids = [], []
    for eid, g in long_df.groupby('eid'):
        wheel_ratio = valid_ratio(g, WHEEL_KEYS)
        dlc_ratio = valid_ratio(g, DLC_KEYS)
        ok_wheel = (np.isnan(wheel_ratio) or wheel_ratio >= min_valid_ratio)
        ok_dlc   = (np.isnan(dlc_ratio)   or dlc_ratio   >= min_valid_ratio)
        keep = bool(ok_wheel and ok_dlc)
        rows.append(dict(eid=eid, wheel_ratio=wheel_ratio, dlc_ratio=dlc_ratio, keep=keep, n_rows=len(g)))
        if keep:
            keep_eids.append(eid)

    qc_summary = pd.DataFrame(rows).sort_values(['keep', 'wheel_ratio', 'dlc_ratio'], ascending=[False, False, False])
    filtered = long_df[long_df['eid'].isin(keep_eids)].copy()

    # final rowwise clean across metrics
    metric_cols = [c for c in filtered.columns if c.startswith(('ave_', 'cv_'))]
    if metric_cols:
        mask = np.ones(len(filtered), dtype=bool)
        for c in metric_cols:
            vals = pd.to_numeric(filtered[c], errors='coerce').values
            mask &= np.isfinite(vals)
        filtered = filtered[mask].copy()

    return filtered, qc_summary

def save_qc_summary(df, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] saved QC summary: {out_path}")


# =====================
# Main
# =====================
def main():
    one = ONE()

    # Load sessions/trials
    sessions = load_sessions(C.DATAPATH)
    trials = load_trials(C.DATAPATH)

    # ---- Pre-filter by video/DLC QC (left & body must meet minimum) ----
    eids_all = sessions['eid'].tolist()
    eids_qc = filter_video_data_multi(
        one, eids_all, cameras=('left', 'body'),
        min_video_qc=VIDEO_QC_MIN, min_dlc_qc=DLC_QC_MIN, require='all'
    )
    print(f"Video/DLC QC kept {len(eids_qc)}/{len(eids_all)} sessions")

    # Restrict to QC-passing sessions
    sessions = sessions[sessions['eid'].isin(eids_qc)].reset_index(drop=True)

    # Compute per-session timecourses
    all_dfs = []
    for eid in tqdm(sessions['eid'].tolist(), desc="Sessions"):
        try:
            per_session_dfs = process_one_session(one, eid, trials)
            all_dfs.extend(per_session_dfs)
        except Exception as e:
            print(f"[warn] session {eid} failed: {e}")

    # RAW long
    raw_long = explode_to_long(all_dfs)
    save_parquet(raw_long, C.DATAPATH / RAW_OUT)

    # QC
    qc_filtered, qc_summary = qc_sessions(raw_long, min_valid_ratio=MIN_VALID_RATIO)
    save_parquet(qc_filtered, C.DATAPATH / QC_OUT)
    save_qc_summary(qc_summary, C.DATAPATH / QC_SUMMARY_CSV)

    kept = int(qc_summary['keep'].sum()) if len(qc_summary) else 0
    total = len(qc_summary)
    print(f"QC kept {kept} / {total} sessions ({(kept/total if total else 0):.1%})")


if __name__ == "__main__":
    main()

# %%
