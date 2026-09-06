# SymmBFN

Generative modeling of crystal structures using **Bayesian Flow Networks (BFN)** with explicit crystallographic symmetry constraints. SymmBFN learns to generate realistic crystal structures that respect space group symmetry by jointly modeling fractional coordinates, atom types, site-symmetries and lattice parameters.

The framework supports two tasks:
- **DNG** (De Novo Generation): unconditional generation of novel crystal structures.
- **CSP** (Crystal Structure Prediction): conditional generation given a composition.

---


## Installation

We recommend [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) for environment management.

```bash
micromamba env create -f env_gpu.yml
micromamba activate symmbfn
pip install -e .
```

### Environment variables

Copy or create a `.env` file in the project root and set the following paths:

```bash
PROJECT_ROOT=/path/to/symmbfn
HYDRA_JOBS=/path/to/hydra/outputs
WABDB_DIR=/path/to/wandb
```

These are loaded automatically at runtime.

---

## Data

Preprocessed datasets are stored under `data/`. Each dataset directory contains CSV splits (`train.csv`, `val.csv`, `test.csv`) with structure representations and target properties. On first use, the dataloader preprocesses structures into cached `.pt` graph tensors.

---

## Training

Training is managed by Hydra. The main entry point is:

```bash
python symmbfn/model/symmbfn_train.py
```

By default this trains a DNG model on MP-20. Common overrides:

```bash
# Train on a different dataset
python symmbfn/model/symmbfn_train.py data=perov

# Train the CSP variant
python symmbfn/model/symmbfn_train.py model=csp

# Adjust loss weights
python symmbfn/model/symmbfn_train.py model.cost_sym=15 model.cost_type=5

# Multi-run hyperparameter sweep (uses joblib launcher)
python symmbfn/model/symmbfn_train.py -m \
  model.bfn.beta1=0.5,0.75,1.0 \
  model.bfn.beta1_sym=1.5,2.0
```

### Key hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `model.bfn.sigma1_coord` | 0.02 | Coordinate noise schedule end-point |
| `model.bfn.sigma1_lattice` | 0.02 | Lattice noise schedule end-point |
| `model.bfn.beta1` | 0.25 | Discrete atom type noise parameter |
| `model.bfn.beta1_sym` | 2.5 | Symmetry noise parameter |
| `model.bfn.sample_steps` | 100 | Steps used during generation |
| `model.cost_coord` | 1 | Coordinate loss weight |
| `model.cost_lattice` | 0.1 | Lattice loss weight |
| `model.cost_type` | 3 | Atom type loss weight |
| `model.cost_sym` | 10 | Symmetry loss weight |
| `optim.optimizer.lr` | 1e-3 | Learning rate |

Training uses `ReduceLROnPlateau` (factor 0.6, patience 100) and early stopping (dataset-specific patience). Checkpoints and WandB logs are written to the paths set in `.env`.

---

## Generation

To generate new crystal structures from a trained model:

```bash
python symmbfn/scripts/generation.py \
  --model_path <path/to/model_dir> \
  --n_samples <number_of_structures> \
  --dist_path <path/to/data/dist.pt>
```

- `--model_path` — directory containing the Hydra run output (subfolders are scanned automatically).
- `--n_samples` — number of structures to generate.
- `--dist_path` — path to the `dist.pt` distribution file for the target dataset (e.g. `data/mp_20/dist.pt`).

---

## Evaluation

### Metrics

```bash
python symmbfn/scripts/compute_metrics.py \
  --root_path <path/to/model_dir> \
  --gt_path <path/to/ground_truth.csv>
```

Reports validity, diversity, and coverage (recall/precision) of generated structures against the test set. Results are written to `eval_metrics.json` inside `--root_path`.

### Energy relaxation

```bash
python symmbfn/scripts/energy_relaxation.py \
  --root_path <path/to/model_dir> \
  --n_samples <number_of_structures>
```

Relaxes generated structures and reports post-relaxation energies.

### CSP generation

```bash
python symmbfn/scripts/evaluate.py --model_path <path/to/model_dir>
```

Runs structure generation conditioned on the test set compositions (CSP task) and saves the predicted structures for downstream analysis.

---

## Project Structure

```
symmbfn/
├── symmbfn/
│   ├── common/          # Constants, data utilities, env helpers
│   ├── data/            # CrystDataset, CrystDataModule
│   ├── model/
│   │   ├── bfn/         # BFN core (bfn_dng.py, bfn_csp.py)
│   │   ├── cspnet/      # CSPNet message-passing architecture
│   │   ├── callbacks/   # Lightning callbacks (EMA, gradient clipping)
│   │   ├── model.py     # CrystalDNG and CrystalCSP Lightning modules
│   │   └── symmbfn_train.py
│   └── scripts/         # generation.py, compute_metrics.py, evaluate.py
├── conf/                # Hydra config tree (data/, model/, optim/, train/, logging/)
├── data/                # Dataset splits and cached graph tensors
├── env_gpu.yml          # Conda environment specification
└── setup.py
```

---

## Acknowledgements

This work is partly based on [SymmCD](https://github.com/SymmCD), [CDVAE](https://github.com/txie-93/cdvae), and [GeoBFN](https://github.com/GeoBFN). We thank the authors for their work and for open-sourcing their code.
