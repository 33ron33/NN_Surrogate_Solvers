# Neural networks as low-cost surrogates for impurity solvers in quantum embedding methods

This repository contains plotting scripts, figure data, and final figure outputs for the paper.

## Paper

- arXiv: [2603.25557](https://arxiv.org/abs/2603.25557)
- DOI: [10.1103/9ndg-shty](https://doi.org/10.1103/9ndg-shty)

## Setup

Create the conda environment:

```bash
conda env create -f NN_Solver.yml
```

Activate the environment:

```bash
conda activate NN_Solver
```

## Repository structure

Each figure is self-contained. Its script, input data, and generated output live in the same figure folder:

```text
figures/
  fig02_green_functions/
    make_fig02_green_functions.py
    data/
    output/
```

The included `.npz`, `.csv`, and text data files were generated from the trained neural-network solver and supporting QMC calculations used in the paper.

## Figures

- `figures/fig01_workflow_architecture/`
- `figures/fig02_green_functions/`
- `figures/fig03_dmft_convergence/`
- `figures/fig04_spinodal_beta25/`
- `figures/fig05_phase_diagram/`
- `figures/fig06_double_occupancy_grid/`
- `figures/fig07_metal_branch_d_z/`
- `figures/fig08_mae_dataset_scaling/`
- `figures/fig09_qmc_acceleration/`
- `figures/fig10_pca_training_coverage/`

## Reproducing a figure

Run the plotting script inside the desired figure folder. For example:

```bash
cd figures/fig03_dmft_convergence
python make_fig03_dmft_convergence.py
```

The regenerated plot is written to that figure folder's `output/` directory.
