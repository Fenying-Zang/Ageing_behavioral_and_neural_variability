# config.py
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


PROJECT_ROOT = Path(__file__).resolve().parent
DATAPATH = PROJECT_ROOT / 'data'
FIGPATH   = PROJECT_ROOT / 'figures'
RESULTSPATH = PROJECT_ROOT / 'results'  
# RESULTSPATH = PROJECT_ROOT / 'results_test'  

PALETTE = {'young': '#78c679', 'old': '#2c7fb8'}
PALETTE5 = ["#bfb4ca", "#9285a3", "#756388", "#5b496e", "#4B3169"]
PALETTE5_2GROUPS = {
    'old':   ['#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c'],
    'young': ['#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45'],
}

COLORS_SWANSON = ['#78c679', 'silver', '#2c7fb8']
COLORS_SWANSON_INVERT = ['#2c7fb8', 'silver', '#78c679']

EVENT_LIST = [
    'stimOn_times',
    'choice',
    'feedback_times',
    'probabilityLeft',
    'firstMovement_times',
    'response_times',  # added to BWM setting
    'feedbackType',
]

ROIS = [
    'MOs', 'ACA', 'CP', 'LS', 'ACB', 'mPFC', 'ORB', 'OLF',
    'VISp+pm', 'SCm', 'MBm', 'PPC', 'CA1', 'DG', 'LP', 'PO'
]

BERYL_NAMES = [
    'VISa', 'VISam', 'VISp', 'VISpm', 'CA1', 'DG', 'PO',
    'LP', 'APN', 'MRN', 'SCm', 'MOs', 'ACAv', 'ACAd',
    'PL', 'ILA', 'ORBm', 'ORBl', 'ORBvl', 'TTd', 'DP',
    'AON', 'ACB', 'CP', 'LSr', 'LSc', 'LSv'
]

TRIAL_TYPE = 'first400'
RT_VARIABLE_NAME = 'response_times_from_stim'    # or 'firstMovement_times_from_stim'
RT_CUTOFF = [0.08, 2.0]                           # 80ms, 2s

AGE2USE = 'age_years'
AGE_GROUP_THRESHOLD = 228.3

SAVE_RESULTS = True

BIN_SIZE = 0.1                      # 
ALIGN_EVENT = 'stim'                # 'stim' | 'move' | 'feedback'
EVENT_EPOCH = [-0.4, 0.8]
SMOOTHING = 'sliding'
SLIDE_KWARGS = {'n_win': 5, 'causal': 1}

CLEAN_RT = True
FIRING_RATE_THRESHOLD = 1
PRESENCE_RATIO_THRESHOLD = 0.95
PIDS_WITHOUT_ILBLSORTOR = ['57edc590-a53d-403c-9aab-d58ee51b6a24', 'daadb3f1-bef2-474e-a659-72922f3fcc5b', '61bb2bcd-37b4-4bcc-8f40-8681009a511a', 'ee2ce090-696a-40f5-8f29-7107339bf08e']

PRE_TIME = 0.0
POST_TIME = 0.26
TOLERANCE = 1e-6
RANDOM_STATE =123

METRICS_WITHOUT_MEANSUB = [
    ('pre_fr', 'mean'),
    ('post_fr', 'mean'),
    ('fr_delta_modulation', 'mean'),
    ('pre_ff', 'mean'),
    ('post_ff', 'mean'),
    ('ff_quench', 'mean'),
    ('ff_quench_modulation', 'mean'),
]

METRICS_WITH_MEANSUB = [
    ('pre_ff', 'mean'),
    ('post_ff', 'mean'),
    ('ff_quench', 'mean'),
]

N_PERMUT_BEHAVIOR = 10000
# N_PERMUT_NEURAL_OMNIBUS = 100
# N_PERMUT_NEURAL_REGIONAL = 100

# N_PERMUT_BEHAVIOR = 10000
N_PERMUT_NEURAL_OMNIBUS = 1000
N_PERMUT_NEURAL_REGIONAL = 1000


# =====will delete later=====
datapath = DATAPATH
figpath = FIGPATH

palette = PALETTE
palette5 = PALETTE5
palette5_2groups = PALETTE5_2GROUPS

colors_swanson = COLORS_SWANSON
colors_swanson_invert = COLORS_SWANSON_INVERT

event_list = EVENT_LIST
ROIs = ROIS
beryl_names = BERYL_NAMES

trial_type = TRIAL_TYPE
rt_variable_name = RT_VARIABLE_NAME
rt_cutoff = RT_CUTOFF

age2use = AGE2USE
age_group_threshold = AGE_GROUP_THRESHOLD

save_results = SAVE_RESULTS
bin_size = BIN_SIZE
align_event = ALIGN_EVENT
event_epoch = EVENT_EPOCH
smoothing = SMOOTHING
slide_kwargs = SLIDE_KWARGS
clean_rt = CLEAN_RT

firing_rate_threshold = FIRING_RATE_THRESHOLD
presence_ratio_threshold = PRESENCE_RATIO_THRESHOLD

pre_time = PRE_TIME
post_time = POST_TIME
tolerance = TOLERANCE

metrics_without_meansub = METRICS_WITHOUT_MEANSUB
metrics_with_meansub = METRICS_WITH_MEANSUB

n_permut_behavior = N_PERMUT_BEHAVIOR
n_permut_neural_omnibus = N_PERMUT_NEURAL_OMNIBUS
n_permut_neural_regional = N_PERMUT_NEURAL_REGIONAL



# ===== Deprecation shim: old --> new =====
import warnings as _warn

_DEPRECATED_ALIASES = {
    "datapath": "DATAPATH",
    "figpath": "FIGPATH",
    "palette": "PALETTE",
    "palette5": "PALETTE5",
    "palette5_2groups": "PALETTE5_2GROUPS",
    "colors_swanson": "COLORS_SWANSON",
    "colors_swanson_invert": "COLORS_SWANSON_INVERT",
    "event_list": "EVENT_LIST",
    "ROIs": "ROIS",
    "beryl_names": "BERYL_NAMES",
    "trial_type": "TRIAL_TYPE",
    "rt_variable_name": "RT_VARIABLE_NAME",
    "rt_cutoff": "RT_CUTOFF",
    "age2use": "AGE2USE",
    "age_group_threshold": "AGE_GROUP_THRESHOLD",
    "save_results": "SAVE_RESULTS",
    "bin_size": "BIN_SIZE",
    "align_event": "ALIGN_EVENT",
    "event_epoch": "EVENT_EPOCH",
    "smoothing": "SMOOTHING",
    "slide_kwargs": "SLIDE_KWARGS",
    "clean_rt": "CLEAN_RT",
    "firing_rate_threshold": "FIRING_RATE_THRESHOLD",
    "presence_ratio_threshold": "PRESENCE_RATIO_THRESHOLD",
    "pre_time": "PRE_TIME",
    "post_time": "POST_TIME",
    "tolerance": "TOLERANCE",
    "metrics_without_meansub": "METRICS_WITHOUT_MEANSUB",
    "metrics_with_meansub": "METRICS_WITH_MEANSUB",
    "n_permut_behavior": "N_PERMUT_BEHAVIOR",
    "n_permut_neural_omnibus": "N_PERMUT_NEURAL_OMNIBUS",
    "n_permut_neural_regional": "N_PERMUT_NEURAL_REGIONAL",
}

def __getattr__(name: str):
    new = _DEPRECATED_ALIASES.get(name)
    if new:
        _warn.warn(
            f"config.{name} 已弃用，请改用 config.{new}",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new]
    raise AttributeError(f"module {__name__} has no attribute {name!r}")


def __dir__():

    return sorted(list(globals().keys()) + list(_DEPRECATED_ALIASES.keys()))

