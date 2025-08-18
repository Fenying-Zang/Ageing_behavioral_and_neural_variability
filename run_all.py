
# run_full.py  —— 从原始数据开始：查询/预处理 -> 计算 -> 统计 -> 出图
import sys, subprocess, config as C

STEPS = [
    "scripts.preprocessing_00_QC_process"

    # 数据准备 / 查询原始数据
    # "scripts.data_00_query_and_cache",
    # 预处理 / 特征计算
    # "scripts.neural_01_compute_metrics",
    # "scripts.neural_02_extract_metrics_summary",
    # 统计 / 置换 / 贝叶斯
    # "scripts.neural_04_stats_neural_yield",
    # 最终出图
    # "scripts.neural_02a_plot_timecourses_summary",
    # "scripts.neural_02b_plot_modulation_timecourses",
    # "scripts.fig2_S1_neural_yield_slice_org",
]

def run(mod):
    print(f"\n=== Running {mod} ===")
    subprocess.check_call([sys.executable, "-m", mod])

if __name__ == "__main__":
    print("Project root:", C.PROJECT_ROOT)
    for mod in STEPS:
        run(mod)
    print("\nEnd-to-end pipeline done")
