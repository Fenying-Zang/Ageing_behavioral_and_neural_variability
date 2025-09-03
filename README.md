# ageing_behavioral_and_neural_variability
*Code and data for analyzing age-related changes in behavioral and neural variability in the IBL visual decision-making task.*

This repository contains the code and data used to analyze **age-related changes in behavioral and neural variability** during a visual decision-making task. We use large-scale extracellular **Neuropixels recordings in behaving mice**, made publicly available by the [International Brain Laboratory (IBL)](https://www.internationalbrainlab.com/).

Fenying Zang, Leiden University, 2025, f.zang@fsw.leidenuniv.nl

---
## Data

- Intermediate files needed for plotting are included in data/ and results/ (via Git LFS, see below).
- Instructions for accessing the public data, along with an online browser, are available at https://docs.internationalbrainlab.org/notebooks_external/data_release_brainwidemap.html and https://www.internationalbrainlab.com/data.
- The release of the newly recorded data is underway, and updates on access will be provided once finalized.

## Installation & Setup

This project builds on the [IBL unified environment](https://github.com/int-brain-lab/iblenv).

### 1. Clone and install the dependencies

```bash
git clone https://github.com/Fenying-Zang/ageing_behavioral_and_neural_variability.git
cd Ageing_behavioral_and_neural_variability
<<<<<<< HEAD
```
=======

>>>>>>> origin/main
# Create and activate a virtual environment using Python 3.10 (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# venv\Scripts\activate   # On Windows

# Install dependencies
<<<<<<< HEAD


```bash
=======
>>>>>>> origin/main
pip install -r requirements.txt

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

## Project Structure

```
ageing_behavioral_and_neural_variability/
├── README.md             # Project overview and setup instructions
├── LICENSE               # MIT License
├── config.py             # Global configuration variables
├── run_figs.py           # Generate all figures
├── run_all_with_R.py     # Full pipeline (requires R; regenerates intermediate data + all figures)
├── run_all_without_R.py  # Full pipeline without R (skips Bayes Factors; regenerates remaining data + all figures)
├── requirements.txt      # Additional Python dependencies (on top of iblenv)
├── data/                 # Intermediate and derived data (tracked via Git LFS)
├── scripts/              # Analysis and plotting code
│   └── utils/            # Shared helper functions
├── figures/              # Generated figures
└── results/              # Output CSVs and summary tables
```

## Reproducing Figures

```bash
# (Optional) Clear previously generated plots
rm -rf figures/*

# Reproduce all manuscript figures from intermediate `.parquet` and `.csv` files
# (located in `data/` and `results/`)
# ⏱ Expected runtime: few minutes
python run_figs.py

# Recompute all intermediate data from raw sources and regenerate figures (requires R)
# (overwrites files in `data/` and `results/`, re-saves figures in `figures/`)
# ⏱ Expected runtime: several hours (depending on machine and data access)
python run_all_with_R.py

# Recompute all intermediate data from raw sources and regenerate figures (without R; skips Bayes Factors)
# (overwrites files in `data/` and `results/`, re-saves figures in `figures/`)
# ⏱ Expected runtime: several hours (depending on machine and data access)
python run_all_without_R.py
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgements

This work builds on the infrastructure, codebase, and datasets shared by the [International Brain Laboratory](https://www.internationalbrainlab.com/). We’re grateful for their open efforts.  
Many thanks to Olivier Winter and Pranav Rai for reviewing the code and suggesting helpful improvements.
