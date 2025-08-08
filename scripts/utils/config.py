# config.py
from pathlib import Path
datapath = Path('../data')
figpath = Path('../figures')

# datapath = '../data'
# figpath = '../figures'

# palette
# palette = ['#78c679', '#2c7fb8']
palette = {'young':'#78c679','old':'#2c7fb8'}
# palette5 = ['#e9e6eb', '#c9c1ce', '#aa9fb3', '#8c7e99', '#6d5d7e']
palette5 = ["#bfb4ca", "#9285a3", "#756388", "#5b496e", "#4B3169"]

palette5_2groups = {
    'old': ['#c6dbef','#9ecae1','#6baed6','#3182bd','#08519c'],
    'young': ['#c7e9c0','#a1d99b','#74c476','#41ab5d','#238b45',]
}

colors_swanson = ['#78c679', 'white', '#2c7fb8']
colors_swanson_invert = [  '#2c7fb8','white','#78c679']

# event_list
event_list = [ 
    'stimOn_times',
    'choice',
    'feedback_times',
    'probabilityLeft',
    'firstMovement_times',
    'response_times',  # added to BWM setting 
    'feedbackType'
]

# ROIs = ['MOs','ACA','CP','LS','ACB','limbic','ORB','olfactory_areas',
#                'visual_cortex','SCm','midbrain_motor','PPC','CA1','DG','LP','PO']
ROIs = ['MOs','ACA','CP','LS','ACB','mPFC','ORB','OLF',
               'VISp+pm','SCm','MBm','PPC','CA1','DG','LP','PO']

beryl_names = ['VISa','VISam','VISp','VISpm','CA1','DG','PO',
     'LP','APN','MRN','SCm','MOs','ACAv','ACAd',
     'PL','ILA','ORBm','ORBl','ORBvl','TTd','DP',
     'AON','ACB','CP','LSr','LSc','LSv']

# beryl_names = np.array(
#     ['VISa','VISam','VISp','VISpm','CA1','DG','PO',
#      'LP','APN','MRN','SCm','MOs','ACAv','ACAd',
#      'PL','ILA','ORBm','ORBl','ORBvl','TTd','DP',
#      'AON','ACB','CP','LSr','LSc','LSv'], #27
#       dtype='<U8')

trial_type = 'first400'
rt_variable_name = 'response_times_from_stim'  # 或 'firstMovement_times_from_stim'
rt_cutoff = [0.08, 2]  # 80ms, 2s - from BWM paper #TODO: discuss the cutoff value
age2use = 'age_years'#'age_years'
age_group_threshold = 228.3 #300

    
save_results = True

bin_size = 0.1 #for now, we use the same window size and step for FFs and frs
align_event = 'stim' #'stim','move', 'feedback'
# event = 'stimOn_times' #movement, feedback
event_epoch = [-0.4, 0.8]
smoothing =  'sliding'
slide_kwargs =  {'n_win': 5, 'causal': 1}
trial_type = 'first400' #'first400'
rt_variable_name = 'response_times_from_stim'#'firstMovement_times_from_stim' or response_times_from_stim
clean_rt = True
rt_cutoff = [0.08, 2]
firing_rate_threshold = 1
presence_ratio_threshold = 0.95

PRE_TIME = 0.0
POST_TIME = 0.26
tolerance = 1e-6  # 设定一个容差值


# metrics_without_meansub = [
#     ('pre_fr', 'mean'),
#     ('post_fr', 'mean'),
#     ('fr_delta_modulation', 'mean'),
#     ('pre_ff', 'median'),
#     ('post_ff', 'median'),
#     ('ff_quench', 'median'),
#     ('ff_quench_modulation', 'mean'),
# ]

# metrics_with_meansub = [
#     ('pre_ff', 'median'),
#     ('post_ff', 'median'),
#     ('ff_quench', 'median'),
# ]
metrics_without_meansub = [
    ('pre_fr', 'mean'),
    ('post_fr', 'mean'),
    ('fr_delta_modulation', 'mean'),
    ('pre_ff', 'mean'),
    ('post_ff', 'mean'),
    ('ff_quench', 'mean'),
    ('ff_quench_modulation', 'mean'),
]

metrics_with_meansub = [
    ('pre_ff', 'mean'),
    ('post_ff', 'mean'),
    ('ff_quench', 'mean'),
]

n_permut_behavior = 10000
n_permut_neural_omnibus=1000
n_permut_neural_regional=2000
