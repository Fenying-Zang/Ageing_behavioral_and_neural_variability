#%%
# run_figs.py  —— 只依赖已经算好的结果/中间表，快速出所有图
import sys
import subprocess
import config as C

STEPS = [
    "scripts.behavior_02_plot_main",#×
    # "scripts.behavior_03b_plot_supp_training",#√
    # "scripts.behavior_04_plot_supp_trial_counts",#√
    # "scripts.behavior_05_plot_supp_choice_bias", ##√permu again, need to save files?
    # "scripts.behavior_06_plot_supp_rt_variations",#√permu again, need to save files?
    # "scripts.behavior_07b_plot_supp_movement_time_courses_scatters",#√ plot show() close
    # "scripts.neural_04_neural_yield_slice_org",#√ plot show() close
    # "scripts.neural_04a_plot_timecourses_slice_org",#√ plot show() close
    # "scripts.neural_04b_plot_modulation_timecourses_slice_org",#√ plot show() close
    # "scripts.neural_05_plot_singleFF_logscatter_slice_org",#√ plot show() close
    "scripts.neural_06_plot_scatters_slice_org", #× Omnibus_BFs_stim_first400_2025_06Aug not found
    # "scripts.neural_06_plot_Swanson_map" #涉及下载one √
]


def run(mod):
    print(f"\n=== Running {mod} ===")
    subprocess.check_call([sys.executable, "-m", mod])


if __name__ == "__main__":

    print("Project root:", C.PROJECT_ROOT)
    for mod in STEPS:
        run(mod)
    print("\nAll figures done! :)") 

# %%
