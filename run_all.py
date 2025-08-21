
# run_full.py  —— 从原始数据开始：查询/预处理 -> 计算 -> 统计 -> 出图
import sys, subprocess, config as C

STEPS = [

    #query data and preprocessing
    "scripts.preprocessing_00_QC_process",
    "scripts.preprocessing_01_generate_merged_tables",

    # prepare data
    "scripts.behavior_01_compute_metrics_permutation",
    "scripts.behavior_03a_compute_training",
    "scripts.behavior_07a_process_movement_time_courses",
    "scripts.neural_01_compute_metrics_time_courses",
    "scripts.neural_02_extract_metrics_summary",
    "scripts.neural_03a_stats_permutation",
    "scripts.neural_03b_stats_BFs",

    #generate figs
    "scripts.behavior_02_plot_main",
    "scripts.behavior_03b_plot_supp_training",
    "scripts.behavior_04_plot_supp_trial_counts",
    "scripts.behavior_05_plot_supp_choice_bias",
    "scripts.behavior_06_plot_supp_rt_variations",
    "scripts.behavior_07b_plot_supp_movement_time_courses_scatters",
    "scripts.neural_04_neural_yield_slice_org",
    "scripts.neural_04a_plot_timecourses_slice_org",
    "scripts.neural_04b_plot_modulation_timecourses_slice_org",
    "scripts.neural_05_plot_singleFF_logscatter_slice_org"
    "scripts.neural_06_plot_scatters_slice_org",
    "scripts.neural_06_plot_Swanson_map"
]


def run(mod):
    print(f"\n=== Running {mod} ===")
    subprocess.check_call([sys.executable, "-m", mod])


if __name__ == "__main__":
    print("Project root:", C.PROJECT_ROOT)
    for mod in STEPS:
        run(mod)
    print("\nEnd-to-end pipeline done")
