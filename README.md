# ARS-PROTACs

ARS-PROTACs is an asymmetric, role-specific deep-learning model for binary PROTAC degradation prediction. It combines SE(3)-encoded 3D structures of the warhead, linker and E3 ligand with ESM-2 protein representations, ligand-conditioned protein processing, role-specific component routing, and component-level molecular fingerprints and physicochemical descriptors.

## Data

The random, cold-target and temporal partitions follow the data splits released with [SE(3)-PROTACs](https://github.com/drugparadigm/SE3-protacs).

Place the split files in `data/`:

```text
data/
├── Random_train.csv   Random_val.csv   Random_test.csv
├── Cold_train.csv     Cold_val.csv     Cold_test_target.csv
├── Cold_test_drug.csv
├── Temporal_train.csv Temporal_val.csv
└── mol2_files/
    ├── warhead_<compound id>.mol2
    ├── linker_<compound id>.mol2
    └── e3_ligand_<compound id>.mol2
```

Each CSV must contain `compound id`, `label`, `target_sequence`, `e3_ligase_sequence`, `warhead_smiles`, `linker_smiles` and `e3_ligase_smiles`. Before training, run [`prepare_data.py`](https://github.com/JunJ-L/ARS-PROTACs/blob/main/prepare_data.py) to generate the component MOL2 files from the SMILES columns:

```bash
python prepare_data.py
```

## Installation

Install PyTorch for your CUDA version, then install the remaining dependencies:

```bash
conda install -c conda-forge rdkit openbabel
pip install torch torch-geometric se3-transformer-pytorch fair-esm
pip install numpy pandas scikit-learn tqdm tensorboard
```

## Training and evaluation

Run from the repository root:

```bash
python main.py
```

The default workflow evaluates random, cold-drug, cold-target and temporal settings. Edit `SPLIT_CONFIGS` in `main.py` to select specific settings.

Training uses seed 42, Adam, learning rate `1e-4`, dropout `0.2`, physical batch size 2, and eight-step gradient accumulation. Logs and checkpoints are written to `log/demo_###/`.

## Citation

Please cite the ARS-PROTACs manuscript when available.

Repository: [JunJ-L/ARS-PROTACs](https://github.com/JunJ-L/ARS-PROTACs)
