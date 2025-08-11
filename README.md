# ageing_behavioral_and_neural_variability

This repository contains the code and data used to analyze age-related changes in behavioral and neural variability during a visual decision-making task. We use large-scale extracellular Neuropixels recordings in behaving mice, made publicly available by the International Brain Laboratory (IBL).

Fenying Zang, Leiden University, 2025, f.zang@fsw.leidenuniv.nl


## Installation & Setup

This project builds on the [IBL brain-wide map project](https://github.com/int-brain-lab/paper-brain-wide-map). We recommend installing the same environment to ensure compatibility:

### 1. Install IBL environment via conda

```bash
git clone https://github.com/int-brain-lab/paper-brain-wide-map.git
cd paper-brain-wide-map
conda env create -f environment.yml
conda activate brainwide
```

### 2. Clone this repository

```bash
cd ..
git clone https://github.com/Fenying-Zang/ageing_behavioral_and_neural_variability.git
cd ageing_behavioral_and_neural_variability
```

> All code and notebooks in this repository are compatible with the `brainwide` conda environment.

## Project Structure

```
ageing_behavioral_and_neural_variability/
├── README.md          # Project overview and instructions
├── LICENSE            # MIT license
├── data/              # Intermediate or derived data files
├── scripts/           # Python scripts for analysis and figures
├── notebooks/         # Jupyter notebooks used to explore data
├── figures/           # Output figures
└── results/           # Output CSVs, summary tables
```

## Reproducing Figures

In the `scripts/` folder, you will find script(s) for each figure in the manuscript.

> Intermediate `.parquet` or `.csv` files required for plotting are provided in the `data/` folder for reproducibility.

## Data

To access the raw IBL dataset, please refer to:

- IBL data access: https://int-brain-lab.github.io/ONE/

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgements

This analysis builds upon the infrastructure, codebase, and datasets provided by [the International Brain Laboratory](https://www.internationalbrainlab.com/).

We gratefully acknowledge their efforts in creating and maintaining the [paper-brain-wide-map](https://github.com/int-brain-lab/paper-brain-wide-map) repository.
