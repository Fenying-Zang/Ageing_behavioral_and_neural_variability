"""
Neural data utility functions.

Functions
---------
- bin_spikes2D        : Bin spike trains into trial x neuron x time matrices.
- cal_presence_ratio  : Compute accurate presence ratio and firing rate for clusters.
- smoothing_sliding   : Sliding-window smoothing of binned spikes.
- enrich_df_new       : Build neuron x conditions- and neuron-level DataFrame with metadata.
- combine_regions     : Map beryl atlas regions into broader groups.
"""

import numpy as np
import pandas as pd
from iblutil.numerical import bincount2D


def bin_spikes2D(spike_times, spike_clusters, cluster_ids, align_times, pre_time, post_time, bin_size, weights=None):
    """
    Bin spike trains into a 3D array aligned to trial events.

    Parameters
    ----------
    spike_times : np.ndarray
        All spike timestamps (s).
    spike_clusters : np.ndarray
        Cluster IDs for each spike in spike_times.
    cluster_ids : np.ndarray
        Unique cluster IDs to include (order defines 2nd dimension).
    align_times : np.ndarray
        Trial-aligned event times to center bins around.
    pre_time : float
        Time before event (s).
    post_time : float
        Time after event (s).
    bin_size : float
        Bin width (s).
    weights : np.ndarray or None
        Optional weights for each spike (e.g. amplitude).

    Returns
    -------
    bins : ndarray, shape (n_trials, n_clusters, n_bins)
        Binned spike counts (or weighted sum).
    tscale : ndarray
        Bin centers relative to event time (s).
    """
    n_bins_pre = int(np.ceil(pre_time / bin_size))
    n_bins_post = int(np.ceil(post_time / bin_size))
    n_bins = n_bins_pre + n_bins_post
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    ts = np.repeat(align_times[:, np.newaxis], tscale.size, axis=1) + tscale
    epoch_idxs = np.searchsorted(spike_times, np.c_[ts[:, 0], ts[:, -1]])
    bins = np.zeros(shape=(align_times.shape[0], cluster_ids.shape[0], n_bins))

    for i, (ep, t) in enumerate(zip(epoch_idxs, ts)):
        xind = (np.floor((spike_times[ep[0]:ep[1]] - t[0]) / bin_size)).astype(np.int64)
        w = weights[ep[0]:ep[1]] if weights is not None else None
        yscale, yind = np.unique(spike_clusters[ep[0]:ep[1]], return_inverse=True)
        nx, ny = [tscale.size, yscale.size]
        ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
        r = np.bincount(ind2d, minlength=nx * ny, weights=w).reshape(ny, nx)

        bs_idxs = np.isin(cluster_ids, yscale)
        bins[i, bs_idxs, :] = r[:, :-1]

    tscale = (tscale[:-1] + tscale[1:]) / 2

    return bins, tscale


def cal_presence_ratio(start_point, end_point, spike_times, spike_clusters, cluster_ids, hist_win=10):
    """
    Computes the presence ratio of spike counts: the number of bins where there is at least one
    spike, over the total number of bins, given a specified bin width.
    Parameters
    ----------
    ts : ndarray
        The spike timestamps from which to compute the presence ratio.
    hist_win : float (optional)
        The time window (in s) to use for computing the presence ratio.
    Returns
    -------
    pr : float
        The presence ratio.
    spks_bins : ndarray
        The number of spks in each bin.
    See Also
    --------
    plot.pres_ratio
    Examples
    --------
    1) Compute the presence ratio for unit 1, given a window of 10 s.
        >>> ts = units_b['times']['1']
        >>> pr, pr_bins = bb.metrics.pres_ratio(ts)
    """
    presence_ratio_bin = bincount2D(spike_times, spike_clusters,
                                xbin=hist_win,
                                ybin=cluster_ids, xlim=[start_point, end_point])[0]
    pr_poi = np.sum(presence_ratio_bin > 0, axis=1) / presence_ratio_bin.shape[1]
    spike_count = np.sum(presence_ratio_bin, axis=1)
    fr_poi = spike_count / (end_point - start_point)
    return pr_poi, fr_poi


def smoothing_sliding(spike_times, spike_clusters, cluster_ids, align_times, align_epoch=(-0.2, 0.5), bin_size=0.1, n_win=5,
                      causal=1):
    """
    Smooth spike trains with a sliding window.

    Parameters
    ----------
    spike_times : np.ndarray
        All spike timestamps (s).
    spike_clusters : np.ndarray
        Cluster IDs for each spike.
    cluster_ids : np.ndarray
        Unique cluster IDs to include.
    align_times : np.ndarray
        Event times to align to.
    align_epoch : tuple, optional
        Time window relative to event.
    bin_size : float, optional
        Bin width (s).
    n_win : int, optional
        Number of sub-bins within each bin for smoothing.
    causal : bool, optional
        If True, enforce causal alignment.

    Returns
    -------
    all_bins : ndarray
        Smoothed spike count array (n_trials x n_clusters x n_bins).
    all_times : ndarray
        Time vector for bins.
    """
    t_shift = bin_size / n_win
    epoch = [align_epoch[0], align_epoch[1]]
    if causal:
        epoch[0] = epoch[0] - (bin_size / 2)

    for w in range(n_win):

        bins, tscale = bin_spikes2D(spike_times, spike_clusters, cluster_ids, (align_times + w * t_shift), np.abs(epoch[0]),
                                    epoch[1] - (w * t_shift), bin_size)
        if w == 0:
            all_bins = bins
            all_times = tscale + w * t_shift
        else:
            all_bins = np.c_[all_bins, bins]
            all_times = np.r_[all_times, tscale + w * t_shift]

    if causal == 1:
        all_times = all_times + bin_size / 2

    sort_idx = np.argsort(all_times)
    all_bins = all_bins[:, :, sort_idx]
    all_times = all_times[sort_idx]

    return all_bins, all_times


def enrich_df_new(conditionsplit=False, k=None, id=None, ff=None, fr=None, time_ff=None, bins_mean=None, bins_std=None, fr_normalized=None,
              pid=None, eid=None,  age_at_recording=None, sex=None, subject=None, sess_date=None, signed_contrast=None,
              n_trials=None, trial_type=None, 
              clusters=None, cluster_idx=None): 
    """
    Build a DataFrame of neuron-level metrics and metadata.

    Parameters
    ----------
    conditionsplit : bool
        Whether to include condition-specific info (e.g. contrast).
    k : int
        Neuron index.
    id : int
        Cluster ID.
    ff : np.ndarray
        Fano factor array [n_neurons x n_timepoints].
    fr : np.ndarray
        Firing rate array [n_neurons x n_timepoints].
    time_ff : np.ndarray
        Timepoints for metrics.
    bins_mean : np.ndarray
        Mean Fano factor across bins.
    bins_std : np.ndarray
        Variance of Fano factor across bins.
    fr_normalized : np.ndarray
        Normalized firing rates.
    pid, eid : str
        Probe and session identifiers.
    age_at_recording : float
        Mouse age at session (days).
    sex : str
        Mouse sex.
    subject : str
        Mouse nickname.
    sess_date : str
        Session date.
    signed_contrast : float, optional
        Trial contrast (if conditionsplit).
    n_trials : int
        Number of trials for this neuron.
    trial_type : str
        Trial type label.
    clusters : pd.DataFrame
        Cluster metadata table.
    cluster_idx : slice or list
        Index into clusters for alignment.

    Returns
    -------
    df_curr : pd.DataFrame
        DataFrame with neuron metrics and metadata.
    """
    #for each neuron/cluster:
    curr = {}
    curr['FFs'] = ff[k,:]
    curr['FF_mean'] = bins_mean[k,:]
    curr['FF_variance'] = bins_std[k,:]

    curr['frs'] = fr[k,:]
    curr['frs_normalized'] = fr_normalized[k,:]
    curr['timepoints'] = time_ff

    df_curr = pd.DataFrame.from_dict(curr) 

    #4.region
    #5.cluster id
    df_curr['cluster_ids'] = id
    df_curr['cluster_region'] = clusters['Beryl_merge'][cluster_idx][k] 
    df_curr['x'] = clusters['x'][cluster_idx][k] 
    df_curr['y'] = clusters['y'][cluster_idx][k] 
    df_curr['z'] = clusters['z'][cluster_idx][k]   

    #2.mouse
    df_curr['mouse_age'] = age_at_recording
    df_curr['mouse_sex'] = sex
    df_curr['mouse_sub_name'] = subject
    # df_curr['mouse_lab']= lab
    #3.session
    df_curr['session_eid'] = eid
    df_curr['session_pid'] = pid
    df_curr['session_date'] = sess_date

    #6.trials
    if conditionsplit:
        df_curr['signed_contrast'] = signed_contrast

    df_curr['n_trials']=n_trials  
    df_curr['trials_type']= trial_type

    return df_curr

# def combine_regions(regions):
#     """
#     Combine all layers of cortex and the dentate gyrus molecular and granular layer
#     Combine VISa and VISam into PPC
#     """
#     remove = ['1', '2', '3', '4', '5', '6a', '6b', '/']
#     for i, region in enumerate(regions):
#         # print(region)
#         # if region[:2] == 'CA':
#         #     continue

#         # for j, char in enumerate(remove):
#         #     regions[i] = regions[i].replace(char, '')
#         if (regions[i] == 'VISa') | (regions[i] == 'VISam'):
#             regions[i] = 'PPC'
#         if (regions[i] == 'VISp') | (regions[i] == 'VISpm'):
#             regions[i] = 'VISp+pm'
#         if (region == 'DG-mo') or (region == 'DG-sg') or (region == 'DG-po'):
#             regions[i] = 'DG'
#         if (region == 'APN') or (region == 'MRN'):
#             regions[i] = 'MBm'

#         if (region == 'ACAv') or (region == 'ACAd'):
#             regions[i] = 'ACA'
#         if (region == 'PL') or (region == 'ILA'):
#             regions[i] = 'mPFC'
#         if (region == 'ORBm') or (region == 'ORBl') or (region == 'ORBvl'):
#             regions[i] = 'ORB'

#         if (region == 'TTd') or (region == 'DP') or (region == 'AON'):
#             regions[i] = 'OLF'
#         if (region == 'LSr') or (region == 'LSc') or (region == 'LSv'):
#             regions[i] = 'LS'

#     return regions

def combine_regions(regions):
    """
    Map fine-grained brain region acronyms into broader groups.

    Parameters
    ----------
    regions : array-like
        Region acronyms (list, ndarray, or Series).

    Returns
    -------
    np.ndarray
        Mapped region names (e.g. VISa/VISam → PPC).
    """
    mapping = {
        'VISa': 'PPC', 'VISam': 'PPC',
        'VISp': 'VISp+pm', 'VISpm': 'VISp+pm',
        'DG-mo': 'DG', 'DG-sg': 'DG', 'DG-po': 'DG',
        'APN': 'MBm', 'MRN': 'MBm',
        'ACAv': 'ACA', 'ACAd': 'ACA',
        'PL': 'mPFC', 'ILA': 'mPFC',
        'ORBm': 'ORB', 'ORBl': 'ORB', 'ORBvl': 'ORB',
        'TTd': 'OLF', 'DP': 'OLF', 'AON': 'OLF',
        'LSr': 'LS', 'LSc': 'LS', 'LSv': 'LS',
    }
    regions = pd.Series(regions) 
    return regions.replace(mapping).to_numpy()
