
"""
run_full.py

Master script to run the **entire pipeline** end-to-end:
    1. Data query and preprocessing
    2. Data preparation and analysis (behavioral + neural)
    3. Figure generation

This ensures all intermediate results and figures can be reproduced
from scratch with a single command.
"""
import sys, subprocess, config as C

# Ordered pipeline steps (modules run sequentially).
# Each step is a standalone script runnable via `python -m`.
STEPS = [

    # --- Query and preprocessing ---
    "scripts.preprocessing_00_QC_process",              # QC filtering
    "scripts.preprocessing_01_generate_merged_tables",  # Merge tables

    # --- Data preparation and analysis ---
    "scripts.behavior_01a_compute_metrics_permutation",  # Behavioral metrics + permutation
    "scripts.behavior_02a_compute_training",             # Training history
    "scripts.neural_01_compute_metrics_time_courses",    # Neural metrics (time courses)
    "scripts.neural_02_extract_metrics_summary",         # Extract summary metrics
    "scripts.neural_03a_stats_permutation",              # Permutation stats
    "scripts.neural_03b_stats_BFs",                      # Bayesian Factors

    # --- Figure generation ---
    "scripts.behavior_01b_plot_main",                    # Main behavior figures
    "scripts.behavior_02b_plot_supp_training",           # Supp: training
    "scripts.behavior_03_plot_supp_trial_counts",        # Supp: trial counts
    "scripts.behavior_04_plot_supp_choice_bias",         # Supp: choice bias
    "scripts.behavior_05_plot_supp_rt_variations",       # Supp: RT variability
    "scripts.neural_04_neural_yield_slice_org",          # Supp: Neural yield
    "scripts.neural_05_plot_timecourses_slice_org",      # Neural time courses
    "scripts.neural_06_plot_modulation_timecourses_slice_org",  # Modulation time courses
    "scripts.neural_07_plot_scatters_slice_org",         # Scatter plots
    "scripts.neural_08_plot_Swanson_map",                # Swanson brain maps
    "scripts.neural_09_plot_singleFF_logscatter_slice_org"  # Supp: Single FF log-log scatter
]

def run(mod):
    print(f"\n=== Running {mod} ===")
    subprocess.check_call([sys.executable, "-m", mod])


if __name__ == "__main__":
    print("Project root:", C.PROJECT_ROOT)
    for mod in STEPS:
        run(mod)
    print("\nEnd-to-end pipeline done")
