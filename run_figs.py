"""
run_figs.py

Master script to generate all behavioral and neural figure panels.

Includes:
    - Sequential execution of all plotting scripts (behavior + neural)
    - Uses subprocess to call each module as `python -m`
    - Ensures reproducibility of full figure pipeline from intermediate results
"""
import sys
import subprocess
import config as C

# List of figure-generating scripts (executed in order).
# Each entry is a module path (string) that can be run with `python -m`
STEPS = [
    "scripts.behavior_01b_plot_main",
    "scripts.behavior_02b_plot_supp_training",
    "scripts.behavior_03_plot_supp_trial_counts",
    "scripts.behavior_04_plot_supp_choice_bias",
    "scripts.behavior_05_plot_supp_rt_variations",
    "scripts.neural_04_neural_yield_slice_org",
    "scripts.neural_05_plot_timecourses_slice_org",
    "scripts.neural_06_plot_modulation_timecourses_slice_org",
    "scripts.neural_07_plot_scatters_slice_org",
    "scripts.neural_08_plot_Swanson_map",
    "scripts.neural_09_plot_singleFF_logscatter_slice_org"

]


def run(mod):
    print(f"\n=== Running {mod} ===")
    subprocess.check_call([sys.executable, "-m", mod])


if __name__ == "__main__":

    print("Project root:", C.PROJECT_ROOT)
    for mod in STEPS:
        run(mod)
    print("\nAll figures done! :)") 
