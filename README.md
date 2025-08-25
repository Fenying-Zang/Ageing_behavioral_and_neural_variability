# ageing_behavioral_and_neural_variability
*Code and data for analyzing age-related changes in behavioral and neural variability in the IBL visual decision-making task.*

This repository contains the code and data used to analyze **age-related changes in behavioral and neural variability** during a visual decision-making task. We use large-scale extracellular **Neuropixels recordings in behaving mice**, made publicly available by the [International Brain Laboratory (IBL)](https://www.internationalbrainlab.com/).

Fenying Zang, Leiden University, 2025, f.zang@fsw.leidenuniv.nl

---

## Installation & Setup

This project builds on the [IBL unified environment](https://github.com/int-brain-lab/iblenv).  
For **reproducibility**, all users should first install the IBL environment and then add the extra dependencies required by this project.

### 1. Install IBL environment via conda

See [IBL unified environment](https://github.com/int-brain-lab/iblenv) for full instruction.
```bash
conda update -n base -c defaults conda
conda create --name iblenv python=3.10 --yes
conda activate iblenv
git clone https://github.com/int-brain-lab/iblapps
pip install --editable iblapps
git clone https://github.com/int-brain-lab/iblenv
cd iblenv
pip install --requirement requirements.txt

```

### 2. Install Git LFS (required for this repo)

This repository uses Git LFS (Large File Storage, an extension to Git, so must have Git itself installed) to store large data files (.parquet, .pqt).
Without Git LFS, these files will appear only as small pointer text files instead of the actual data.
- Install Git LFS

  - macOS (Homebrew):

    ```bash
    brew install git-lfs
    ```

  - Windows: Download the installer from [git-lfs.com](https://git-lfs.com/) 

- Initialize Git LFS (only once per machine)

```bash
git lfs install
```

### 3. Clone this repository

```bash
cd ..
git clone https://github.com/Fenying-Zang/ageing_behavioral_and_neural_variability.git
cd ageing_behavioral_and_neural_variability
```

### 4. Install additional dependencies for this project
After activating iblenv and cloning this repository, install the extra dependencies:

```bash
pip install -r requirements.txt

```

## Project Structure

```
ageing_behavioral_and_neural_variability/
├── README.md          # Project overview and instructions
├── LICENSE            # MIT license
├── config.py          # Global variables
├── run_figs.py        # Run this script to generate all figures
├── run_all.py         # Run this script to  regenerate intermediate data from scratch all figures
├── requirements.txt   # Extra Python deps on top of iblenv
├── data/              # Intermediate or derived data (tracked via Git LFS)
├── scripts/           # Python scripts for analysis and plotting
│   └── utils/         # Shared helper functions
├── figures_test/      # Output figures
└── results/           # Output CSVs, summary tables
```

## Reproducing Figures

```bash
# (Optional) Clear previously generated plots
rm -rf figures_test/*

# Reproduce all manuscript figures from intermediate `.parquet` and `.csv` files
# (located in `data/` and `results/`)
# ⏱ Expected runtime: few minutes
python run_figs.py

# To recompute all intermediate data from raw sources and regenerate figures
# (rebuilds `results/` and re-saves figures into `figures_test/`)
# ⏱ Expected runtime: several hours (depending on machine and data access)
python run_all.py
```

## Data

- Intermediate files needed for plotting are included in data/ and results/ (via Git LFS).
- To access the raw IBL dataset, please refer to: https://int-brain-lab.github.io/ONE/

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgements

This analysis builds upon the infrastructure, codebase, and datasets provided by [the International Brain Laboratory](https://www.internationalbrainlab.com/). We gratefully acknowledge their efforts.
