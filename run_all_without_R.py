"""
run_full_without_R.py

Master script to run the **pipeline** (skipping Bayes Factors):
    1. Data query and preprocessing
    2. Data preparation and analysis (behavioral + neural)
    3. Figure generation

This ensures all intermediate results (skipping Bayes Factors) and figures can be reproduced
from scratch with a single command.
"""
import sys, subprocess, config as C

# Ordered pipeline steps (modules run sequentially).
# Each step is a standalone script runnable via `python -m`.
STEPS = [

    # --- Query and preprocessing ---
    "aging_variability.preprocessing.preprocessing_00_QC_process",              # QC filtering √
    # "aging_variability.preprocessing.preprocessing_01_generate_merged_tables",  # NOTE: Merge trials tables — optional; skip to use pre-merged tables and save time

    # --- Data preparation and analysis ---
    "aging_variability.behavior.behavior_01a_compute_metrics_permutation",  # Behavioral metrics + permutation √
    "aging_variability.behavior.behavior_02a_compute_training",             # Training history √
    "aging_variability.neural.neural_01_compute_metrics_time_courses",    # Neural metrics (time courses)
    "aging_variability.neural.neural_02_extract_metrics_summary",         # Extract summary metrics
    "aging_variability.neural.neural_03a_stats_permutation",              # Permutation stats
    # "aging_variability.neural.neural_03b_stats_BFs",                      # Bayesian Factors -- NOTE: without R, let's skip it here

    # --- Figure generation ---
    "aging_variability.behavior.behavior_01b_plot_main",                    # Main behavior figures
    "aging_variability.behavior.behavior_02b_plot_supp_training",           # Supp: training
    "aging_variability.behavior.behavior_03_plot_supp_trial_counts",        # Supp: trial counts
    "aging_variability.behavior.behavior_04_plot_supp_choice_bias",         # Supp: choice bias
    "aging_variability.behavior.behavior_05_plot_supp_rt_variations",       # Supp: RT variability
    "aging_variability.neural.neural_04_neural_yield_slice_org",          # Supp: Neural yield
    "aging_variability.neural.neural_05_plot_timecourses_slice_org",      # Neural time courses
    "aging_variability.neural.neural_06_plot_modulation_timecourses_slice_org",  # Modulation time courses
    "aging_variability.neural.neural_07_plot_scatters_slice_org",         # Scatter plots
    "aging_variability.neural.neural_08_plot_Swanson_map",                # Swanson brain maps
    "aging_variability.neural.neural_09_plot_singleFF_logscatter_slice_org"  # Supp: Single FF log-log scatter
]

def run(mod):
    print(f"\n=== Running {mod} ===")
    subprocess.check_call([sys.executable, "-m", mod])


if __name__ == "__main__":
    print("Project root:", C.PROJECT_ROOT)
    for mod in STEPS:
        run(mod)
    print("\nPipeline done! Congrats!")
