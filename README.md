# neural_variability_aging

This repository contains the code and data used to analyze age-related changes in neural variability during a visual decision-making task. We use large-scale extracellular Neuropixels recordings in awake, behaving mice, made publicly available by the International Brain Laboratory (IBL).

Our large-scale survey of single-neuron functioning shows that there is no simple, global increase in neural variability in older animals.

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
git clone https://github.com/YOUR_USERNAME/neural_variability_aging.git
cd neural_variability_aging
```

> All code and notebooks in this repository are compatible with the `brainwide` conda environment.

## Project Structure

```
neural_variability_aging/
├── README.md          # Project overview and instructions
├── LICENSE            # MIT license
├── data/              # Intermediate or derived data files
├── scripts/           # Python scripts for analysis
├── notebooks/         # Jupyter notebooks used to generate figures
├── figures/           # Output figures
└── results/           # Output CSVs, summary tables, models
```

## Reproducing Figures

Each Jupyter notebook in `notebooks/` corresponds to one or more figures in the manuscript.

> Intermediate `.pkl` or `.csv` files required for plotting are provided in the `data/` folder for reproducibility.

## Data

This project does **not** include raw data. To access the full IBL dataset, please refer to:

- IBL data access: https://int-brain-lab.github.io/ONE/
- ONE light datasets used here are downloaded via scripts in the original [paper-brain-wide-map](https://github.com/int-brain-lab/paper-brain-wide-map)

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgements

This analysis builds upon the infrastructure, codebase, and datasets provided by [the International Brain Laboratory](https://www.internationalbrainlab.com/).

We gratefully acknowledge their efforts in creating and maintaining the [paper-brain-wide-map](https://github.com/int-brain-lab/paper-brain-wide-map) repository.
