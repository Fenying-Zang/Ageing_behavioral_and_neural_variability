"""
Plotting utilities.

Includes
--------
- figure_style / set_seaborn : Standardized plot styles (IBL-like, print-friendly).
- create_slice_org_axes      : Slice-organized brain region layout for multi-panel plots.
- plot_psychometric          : Fit + plot psychometric function across contrasts.
- plot_chronometric          : Plot chronometric (RT) function across contrasts.
- break_xaxis                : Draw discontinuous axis markers.
- add_n                      : Annotate plots with subject/trial counts.
- num_star / num_star_001     : Map p-values to star annotations.
- map_p_value                : Convert p-values to formatted string.
- plot_permut_test           : Visualize permutation test distributions.
- format_bf_annotation       : Build annotation text with β, p_perm, and Bayes factor.
- add_window_label           : Draw labeled horizontal bars for analysis windows.
"""
import seaborn as sns
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ibl_style.utils import get_coords, MM_TO_INCH, double_column_fig
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.transforms as mtransforms


def figure_style():
    """
    Apply IBL-style plotting defaults for small scientific figures.
    - Uses Arial font, thin axes, small ticks.
    - Intended for multi-panel journal figures.
    """
    sns.set_theme(style="ticks", context="paper",
            rc={"font.size": 7,
                "axes.titlesize": 8,
                "axes.labelsize": 7,
                "axes.linewidth": 0.5,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "legend.title_fontsize": 7,
                "lines.linewidth": 1,
                "lines.markersize": 4,
                "xtick.labelsize": 6,
                "ytick.labelsize": 6,
                "savefig.transparent": False,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.minor.width": 0.5,
                "ytick.minor.width": 0.5,
                "axes.labelcolor": "black",
                "text.color": "black",
                "xtick.color": "black",
                "ytick.color": "black",
                "axes.edgecolor": "black",
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.family'] = 'Arial'


# def set_seaborn():
#     """
#     Apply seaborn-based plotting style for print-friendly figures.
#     Larger fonts than figure_style(), for slides/posters.
#     """
#     sns.set_theme(
#         style="ticks", context="paper",
#         font="Arial",
#         rc={
#             "font.size": 9,
#             "axes.titlesize": 9,
#             "axes.labelsize": 9,
#             "lines.linewidth": 1,
#             "xtick.labelsize": 7,
#             "ytick.labelsize": 7,
#             "savefig.transparent": False,
#             "xtick.major.size": 2.5,
#             "ytick.major.size": 2.5,
#             "xtick.minor.size": 2,
#             "ytick.minor.size": 2,
#             "axes.labelcolor": "black",
#             "text.color": "black",
#             "xtick.color": "black",
#             "ytick.color": "black",
#             "axes.edgecolor": "black",
#         }
#     )
#     matplotlib.rcParams['pdf.fonttype'] = 42
#     matplotlib.rcParams['ps.fonttype'] = 42
#     matplotlib.rcParams['font.family'] = 'Arial'


def create_slice_org_axes(fg, MM_TO_INCH, fig=None):
    """
    Create a slice-organized layout of brain regions (IBL atlas standard).

    Parameters
    ----------
    fg : module
        Figure grid utility (e.g. figrid).
    MM_TO_INCH : float
        Conversion factor from mm to inches.
    fig : matplotlib.Figure or None
        If None, create a new double-column figure.

    Returns
    -------
    fig : matplotlib.Figure
    axs : dict
        Mapping of region name → Axes object.
    """
    if fig is None:
        fig = double_column_fig()

    width, height = fig.get_size_inches() / MM_TO_INCH

    xspans = get_coords(width, ratios=[1, 1, 1, 1], space=20, pad=5, span=(0, 1))
    yspans = get_coords(height, ratios=[1, 1, 1, 1, 1], space=10, pad=5, span=(0, 0.6))

    layout = {
        'MOs': (0, 0), 'ACA': (0, 1), 'CP': (0, 2), 'LS': (0, 3), 'ACB': (0, 4),
        'mPFC': (1, 2), 'ORB': (1, 3), 'OLF': (1, 4),
        'VISp+pm': (2, 2), 'SCm': (2, 3), 'MBm': (2, 4),
        'PPC': (3, 0), 'CA1': (3, 1), 'DG': (3, 2), 'LP': (3, 3), 'PO': (3, 4),
    }

    axs = {
        region: fg.place_axes_on_grid(fig, xspan=xspans[xi], yspan=yspans[yi])
        for region, (xi, yi) in layout.items()
    }

    return fig, axs


def plot_psychometric(x, y, subj, **kwargs):
    """
    Fit and plot psychometric curve for grouped data.

    Parameters
    ----------
    x : array-like
        Signed contrasts (%).
    y : array-like
        Choices (0/1).
    subj : array-like
        Subject/session IDs for grouping.
    kwargs : dict
        Passed to seaborn.lineplot.

    Notes
    -----
    - Fits erf_psycho_2gammas via psychofit.
    - Handles "broken" x-axis when 0% contrast present.
    """
    # import brainbox.behavior.pyschofit as psy
    import psychofit as psy

    # summary stats - average psychfunc over observers
    df = pd.DataFrame({'signed_contrast': x, 'choice': y,
                       'choice2': y, 'subject_nickname': subj})
    df2 = df.groupby(['signed_contrast', 'subject_nickname']).agg(
        {'choice2': 'count', 'choice': 'mean'}).reset_index()
    df2.rename(columns={"choice2": "ntrials",
                        "choice": "fraction"}, inplace=True)
    df2 = df2.groupby(['signed_contrast'])[['ntrials', 'fraction']].mean().reset_index()
    #df2 = df2[['signed_contrast', 'ntrials', 'fraction']]

    # only 'break' the x-axis and remove 50% contrast when 0% is present
    # print(df2.signed_contrast.unique())
    if 0. in df2.signed_contrast.values:
        brokenXaxis = True
    else:
        brokenXaxis = False

    # fit psychfunc
    pars, L = psy.mle_fit_psycho(df2.transpose().values,  # extract the data from the df
                                 P_model='erf_psycho_2gammas',
                                 parstart=np.array(
                                     [0, 20., 0.05, 0.05]),
                                 parmin=np.array(
                                     [df2['signed_contrast'].min(), 5, 0., 0.]),
                                 parmax=np.array([df2['signed_contrast'].max(), 40., 1, 1]))

    if brokenXaxis:
        # plot psychfunc
        g = sns.lineplot(x=np.arange(-27, 27),
                         y=psy.erf_psycho_2gammas(pars, np.arange(-27, 27)), **kwargs)

        # plot psychfunc: -100, +100
        sns.lineplot(x=np.arange(-36, -31),
                     y=psy.erf_psycho_2gammas(pars, np.arange(-103, -98)), **kwargs)
        sns.lineplot(x=np.arange(31, 36),
                     y=psy.erf_psycho_2gammas(pars, np.arange(98, 103)), **kwargs)

        # if there are any points at -50, 50 left, remove those
        if 50 in df.signed_contrast.values or -50 in df.signed_contrast.values:
            df.drop(df[(df['signed_contrast'] == -50.) | (df['signed_contrast'] == 50)].index,
                    inplace=True)

        # now break the x-axis
        df['signed_contrast'] = df['signed_contrast'].replace(-100, -35)
        df['signed_contrast'] = df['signed_contrast'].replace(100, 35)

    else:
        # plot psychfunc
        g = sns.lineplot(x=np.arange(-103, 103),
                         y=psy.erf_psycho_2gammas(pars, np.arange(-103, 103)), **kwargs)

    df3 = df.groupby(['signed_contrast', 'subject_nickname']).agg(
        {'choice2': 'count', 'choice': 'mean'}).reset_index()

    # plot datapoints with errorbars on top
    if df['subject_nickname'].nunique() > 1:
        # put the kwargs into a merged dict, so that overriding does not cause an error
        sns.lineplot(x=df3['signed_contrast'], y=df3['choice'],
                     **{**{'err_style':"bars",
                     'linewidth':0, 'linestyle':'None', 'mew':0.5,
                     'marker':'o', 'errorbar':('ci', 95)}, **kwargs})

    if brokenXaxis:
        g.set_xticks([-35, -25, -12.5, 0, 12.5, 25, 35])
        g.set_xticklabels(['-100', '-25', '-12.5', '0', '12.5', '25', '100'],
                          size='small', rotation=60)
        g.set_xlim([-40, 40])
        break_xaxis()

    else:
        g.set_xticks([-100, -50, 0, 50, 100])
        g.set_xticklabels(['-100', '-50', '0', '50', '100'],
                          size='small', rotation=60)
        g.set_xlim([-110, 110])

    g.set_ylim([0, 1.02])
    g.set_yticks([0, 0.25, 0.5, 0.75, 1])
    g.set_yticklabels(['0', '25', '50', '75', '100'])


def plot_chronometric(x, y, subj, estimator='median', **kwargs):
    """
    Plot chronometric curve (RT vs contrast) with error bars.

    Parameters
    ----------
    x : array-like
        Signed contrasts.
    y : array-like
        Reaction times (s).
    subj : array-like
        Subject/session IDs.
    estimator : str or function
        Aggregation method (e.g., 'median').
    kwargs : dict
        Passed to seaborn.lineplot.
    """
    df = pd.DataFrame(
        {'signed_contrast': x, 'rt': y, 'subject_nickname': subj})
    df.dropna(inplace=True)  # ignore NaN RTs
    df2 = df.groupby(['signed_contrast', 'subject_nickname']
                     ).agg({'rt': estimator}).reset_index()
    # df2 = df2.groupby(['signed_contrast']).mean().reset_index()
    df2 = df2[['signed_contrast', 'rt', 'subject_nickname']]

    # only 'break' the x-axis and remove 50% contrast when 0% is present
    # print(df2.signed_contrast.unique())
    if 0. in df2.signed_contrast.values:
        brokenXaxis = True

        df2['signed_contrast'] = df2['signed_contrast'].replace(-100, -35)
        df2['signed_contrast'] = df2['signed_contrast'].replace(100, 35)
        df2 = df2.loc[np.abs(df2.signed_contrast) != 50, :] # remove those

    else:
        brokenXaxis = False

    ax = sns.lineplot(x='signed_contrast', y='rt', err_style="bars", mew=0.5,
                      errorbar=('ci', 95), data=df2, **kwargs)

    # all the points
    if df['subject_nickname'].nunique() > 1:
        sns.lineplot(
            x='signed_contrast',
            y='rt',
            data=df2,
            **{**{'err_style':"bars",
                     'linewidth':0, 'linestyle':'None', 'mew':0.5,
                     'marker':'o', 'errorbar':('ci', 95)}, **kwargs})

    if brokenXaxis:
        ax.set_xticks([-35, -25, -12.5, 0, 12.5, 25, 35])
        ax.set_xticklabels(['-100', '-25', '-12.5', '0', '12.5', '25', '100'],
                          size='small', rotation=60)
        ax.set_xlim([-40, 40])
        break_xaxis()
        ax.set_ylim([0, df2['rt'].max()*1.1])

    else:
        ax.set_xticks([-100, -50, 0, 50, 100])
        ax.set_xticklabels(['-100', '-50', '0', '50', '100'],
                          size='small', rotation=60)
        ax.set_xlim([-110, 110])


def break_xaxis(y=0, **kwargs):
    """
    Draw visual markers for discontinuous x-axis (hacky overlay).
    Places small // markers near ±30.
    """
    # axisgate: show axis discontinuities with a quick hack
    # https://twitter.com/StevenDakin/status/1313744930246811653?s=19
    # first, white square for discontinuous axis
    plt.text(-30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')
    plt.text(30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')

    # put little dashes to cut axes
    plt.text(-30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=6, fontweight='bold')
    plt.text(30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=6, fontweight='bold')


# def add_n(x, y, sj, **kwargs):
#     """
#     Add annotation with number of subjects and trials.

#     Parameters
#     ----------
#     x, y : array-like
#         Data arrays.
#     sj : array-like
#         Subject IDs.
#     """
#     df = pd.DataFrame({'signed_contrast': x, 'choice': y,
#                        'choice2': y, 'subject_nickname': sj})

#     # ADD TEXT ABOUT NUMBER OF ANIMALS AND TRIALS
#     plt.text(
#         15,
#         0.2,
#         '%d mice, %d trials' %
#         (df.subject_nickname.nunique(),
#          df.choice.count()),
#         fontweight='normal',
#         fontsize=6,
#         color='k')


def num_star(pvalue):
    """Return significance stars (with p-value threshold in text)."""
    if pvalue < 0.0001:
        stars = 'p < 0.0001 ****'
    elif pvalue < 0.001:
        stars = 'p < 0.001 ***'
    elif pvalue < 0.01:
        stars = 'p < 0.01 **'
    elif pvalue < 0.05:
        stars = 'p < 0.05 *'
    else:
        stars = 'n.s.'
    return stars


def map_p_value(pvalue):
    """Convert numeric p-value to formatted string (e.g., ' < 0.01')."""
    if pvalue < 0.0001:
        map_p = ' < 0.0001'
    elif pvalue < 0.001:
        map_p = ' < 0.001'
    elif pvalue < 0.01:
        map_p = ' < 0.01'
    else:
        map_p = f" = {pvalue:.3f}"
    return map_p


def plot_permut_test(null_dist, observed_val, p, mark_p=None, metric=None, save_path=None, show=True, region=None):
    """
    Plot histogram of null distribution with observed value.

    Parameters
    ----------
    null_dist : array-like
        Null distribution from permutations.
    observed_val : float
        Observed test statistic.
    p : float
        Permutation p-value.
    mark_p : float or None
        Optional threshold (e.g. 0.95).
    metric : str
        Name of the metric for labeling.
    save_path : str or Path or None
        If provided, save figure to this path.
    show : bool
        If True, display figure.
    region : str or None
        Optional region name for figure title.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    n, bins, patches = ax.hist(null_dist, bins=25, color='gray', edgecolor='white')

    # Plot the observed value
    ax.plot(observed_val, np.max(n) / 20, '*r', markersize=12, label="Observed β")
    ax.axvline(np.mean(null_dist), color='k', label="Expected β (mean of null)")

    if mark_p is not None:
        critical_point = np.sort(null_dist)[int((1 - mark_p) * len(null_dist))]
        ax.axvline(critical_point, color='r', linestyle='--', label=f"Critical @ p={mark_p}")
        print(f"Critical value at p={mark_p:.2f}: {critical_point:.4f}")

    # Labeling
    ax.set_xlabel('β (age)', fontsize=16)
    ax.set_ylabel('Permutation count', fontsize=16)
    ax.set_title(metric or f"Permutation test\np = {p:.4f}", fontsize=18)
    ax.tick_params(axis='both', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=13)

    plt.tight_layout()

    # Save if needed
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if region is not None:
            save_name = Path("../figures") / f"{metric}_omnibus_permtest.png"
        else:
            save_name = Path("../figures") / f"{metric}_{region}_permtest.png"

        fig.savefig(save_name, dpi=300)
        print(f"[Saved] {save_path}")

    # if show:
    #     #plt.show()()
    # else:
    #     plt.close(fig)

    return fig, ax



def format_bf_annotation(beta, p_perm, BF10, BF_conclusion, beta_label="age", big_bf=100):
    """
    Build the multiline annotation string used in scatter panels.

    Parameters
    ----------
    beta : float
    p_perm : float
    BF10 : float
    BF_conclusion : str
    beta_label : str, default "age"
        LaTeX subscript label for beta, e.g., 'age'.
    big_bf : float, default 100
        Threshold for using '> big_bf' instead of a numeric BF.

    Returns
    -------
    str : formatted annotation string with two lines.
    """
    mapped = map_p_value(p_perm)  # uses  existing helper
    # BF line: "> 100" if big enough, else numeric
    if np.isfinite(BF10) and BF10 > big_bf:
        bf_str = r"$BF_{\mathrm{10}} > " + f"{int(big_bf)}" + r", $"
    else:
        bf_str = r"$BF_{\mathrm{10}} = " + f"{BF10:.3f}" + r", $"

    if np.abs(beta) < 0.001:
        txt = (
            r" $\beta_{\mathrm{" + beta_label + r"}} < " + "0.001, $" +
            r"$p_{\mathrm{perm}} " + f"{mapped}" + r"$" +
            "\n" + bf_str + f" {BF_conclusion}"
        )
    else:
        txt = (
            r" $\beta_{\mathrm{" + beta_label + r"}} = " + f"{beta:.3f}, $" +
            r"$p_{\mathrm{perm}} " + f"{mapped}" + r"$" +
            "\n" + bf_str + f" {BF_conclusion}"
        )  

    return txt



def add_window_label(ax, x_start, x_end, label, *,
                     location='outside',               
                     line_pad=0.02,                    
                     text_pad=0.015,                   
                     lw=1, fontsize=7):
    """
    Draw a horizontal line above/below an interval and annotate it.

    Parameters
    ----------
    ax : matplotlib.Axes
    x_start, x_end : float
        Interval in data coordinates.
    label : str
        Window label.
    location : {'inside','outside'}
        Place inside or above axis.
    """
    # y：inside:1 - line_pad，outside:1 + line_pad
    if location == 'inside':
        y_line = 1.0 - line_pad
    else:
        y_line = 1.0 + line_pad

    # x -> data, y -> axes
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    ax.plot([x_start, x_end], [y_line, y_line],
            transform=trans, color='k', lw=lw, clip_on=False)

    # txt
    label_fmt = rf'$\it{{{label}}}$'
    ax.text((x_start + x_end)/2, y_line + (text_pad if location=='outside' else -text_pad),
            label_fmt, transform=trans, ha='center',
            va=('bottom' if location=='outside' else 'top'),
            fontsize=fontsize, clip_on=False)
