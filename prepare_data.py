"""
mol2 generation
=================================================================
Generate 3D mol2 structures for the three molecular components of each compound
(warhead / linker / e3_ligand) for use by the SE(3)-Transformer encoders.

Workflow: data/*.csv (with SMILES) --> Open Babel make3D --> data/mol2_files/
Output filenames must match the naming convention used by dataset.PROTACDataset:
    data/mol2_files/warhead_{compound id}.mol2
    data/mol2_files/linker_{compound id}.mol2
    data/mol2_files/e3_ligand_{compound id}.mol2

Usage: python prepare_data.py
Run this script from the deployment root containing data/. Existing mol2 files are skipped.
"""

import os
import pandas as pd
from openbabel import pybel, openbabel

openbabel.obErrorLog.SetOutputLevel(0)

# CSV files from the Random, Cold, and Temporal splits; their union covers all compound IDs.
CSV_FILES = [
    "data/Random_train.csv",
    "data/Random_val.csv",
    "data/Random_test.csv",
    "data/Cold_train.csv",
    "data/Cold_val.csv",
    "data/Cold_test_target.csv",
    "data/Cold_test_drug.csv",
    "data/Temporal_train.csv",
    "data/Temporal_val.csv",
    "data/Temporal_test.csv",
]

OUTPUT_DIR = "data/mol2_files"

# Component column name -> mol2 filename prefix (must match the dataset reader).
COMPONENTS = {
    "warhead_smiles": "warhead",
    "linker_smiles": "linker",
    "e3_ligase_smiles": "e3_ligand",
}


def conversion(smiles, prefix, compound_id):
    """Convert one SMILES string to a 3D mol2 file in OUTPUT_DIR, skipping existing files."""
    output_path = os.path.join(OUTPUT_DIR, f"{prefix}_{compound_id}.mol2")
    if os.path.exists(output_path):
        return "skip"
    if not smiles or str(smiles) == "nan":
        return "empty"
    try:
        mol = pybel.readstring("smi", str(smiles))
        mol.make3D()
        mol.write("mol2", output_path, overwrite=True)
        return "ok"
    except Exception as e:
        print(f"[FAIL] {prefix}_{compound_id}: {e}")
        return "fail"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect compound ID -> three-component SMILES mappings without overwriting earlier entries.
    records = {}
    for csv_path in CSV_FILES:
        if not os.path.exists(csv_path):
            print(f"[SKIP CSV] not found: {csv_path}")
            continue
        # utf-8-sig removes a possible BOM so the 'compound id' column name remains valid.
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        for _, row in df.iterrows():
            cid = row["compound id"]
            if cid not in records:
                records[cid] = {c: row.get(c, "") for c in COMPONENTS}

    print(f"Total unique compounds: {len(records)}")

    stats = {"ok": 0, "skip": 0, "empty": 0, "fail": 0}
    for cid, smis in records.items():
        for col, prefix in COMPONENTS.items():
            stats[conversion(smis.get(col, ""), prefix, cid)] += 1

    print(
        "Done. "
        f"ok={stats['ok']}, skip(existing)={stats['skip']}, "
        f"empty={stats['empty']}, fail={stats['fail']}"
    )


if __name__ == "__main__":
    main()
