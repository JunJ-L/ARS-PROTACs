import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data, Batch
from utils import ESMEmbedder
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

LIGAND_ATOM_TYPE = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
FASTA_CHAR = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
              "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
              "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
              "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}
EDGE_ATTR = {'1': 1, '2': 2, '3': 3, 'ar': 4, 'am': 5}


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind.get(ch, smi_ch_ind['X'])
    return X


def tokenize_protein_sequence(sequence):
    tokens = [FASTA_CHAR.get(ch, FASTA_CHAR['X']) for ch in sequence]
    return torch.tensor(tokens, dtype=torch.long)

def mol2graph(path, ATOM_TYPE):
    with open(path) as f:
        lines = f.readlines()
    try:
        atom_end_line = lines.index('@<TRIPOS>UNITY_ATOM_ATTR\n')
    except ValueError:
        atom_end_line = lines.index('@<TRIPOS>BOND\n')

    atom_lines = lines[lines.index('@<TRIPOS>ATOM\n') + 1:atom_end_line]
    bond_lines = lines[lines.index('@<TRIPOS>BOND\n') + 1:]
    atoms = []
    positions = []
    for atom in atom_lines:
        ele = atom.split()[5].split('.')[0]
        atoms.append(ATOM_TYPE.index(ele)
                     if ele in ATOM_TYPE
                     else len(ATOM_TYPE))
        positions.append([eval(atom.split()[2]), eval(atom.split()[3]), eval(atom.split()[4])])
    edge_1 = [int(i.split()[1]) - 1 for i in bond_lines]
    edge_2 = [int(i.split()[2]) - 1 for i in bond_lines]
    edge_attr = [EDGE_ATTR[i.split()[3]] for i in bond_lines]
    x = torch.tensor(atoms)
    edge_idx = torch.tensor([edge_1 + edge_2, edge_2 + edge_1])
    edge_attr = torch.tensor(edge_attr + edge_attr)

    positions = torch.tensor(positions)
    tdEdge = to_scipy_sparse_matrix(edge_idx, edge_attr).todense()
    tdEdge = torch.from_numpy(np.array(tdEdge, dtype=np.float32).flatten())
    graph = Data(x=x, pos=positions, edge=tdEdge)

    return graph
            
def compute_mol_descriptors(smiles, fp_dim=128):
    """Compute a Morgan fingerprint and six physicochemical descriptors for one molecule."""
    mol = Chem.MolFromSmiles(smiles) if (smiles and str(smiles) != 'nan') else None
    if mol is None:
        return np.zeros(fp_dim + 6, dtype=np.float32)
    try:
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_dim)
        fp = mfpgen.GetFingerprint(mol)
        fp_array = np.array(fp, dtype=np.float32)
        physchem = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
        ], dtype=np.float32)
        return np.concatenate([fp_array, physchem])
    except Exception:
        return np.zeros(fp_dim + 6, dtype=np.float32)


class PROTACDataset(Dataset):
    def __init__(self, data_dir, clean_data, geometric=False, sequence=False, use_descriptors=False, fp_dim=128, desc_mode="all",
                 esm_model_name='esm2_t6_8M_UR50D', esm_repr_layer=6):
        self.data_dir = data_dir
        self.data = pd.read_csv(clean_data)
        self.geometric = geometric
        self.sequnce = sequence
        self.use_descriptors = use_descriptors
        self.fp_dim = fp_dim
        self.desc_mode = desc_mode  # "all": W+L+E3(402d), "WaE3": W+E3(268d)
        self.esm = ESMEmbedder(model_name=esm_model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.esm.repr_layer = esm_repr_layer


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        compound_id, target_sequence, e3_ligase_sequence, label = row['compound id'], row['target_sequence'], row['e3_ligase_sequence'], row['label']
        
        self.label = int(label)
        
        warhead_file = self.data_dir + 'warhead_{}.mol2'.format(compound_id)
        ligase_ligand_file = self.data_dir + 'e3_ligand_{}.mol2'.format(compound_id)
        linker_file = self.data_dir + 'linker_{}.mol2'.format(compound_id)

        warhead = mol2graph(warhead_file, LIGAND_ATOM_TYPE)
        ligase_ligand = mol2graph(ligase_ligand_file, LIGAND_ATOM_TYPE)
        linker = mol2graph(linker_file, LIGAND_ATOM_TYPE)

        target_tokens = tokenize_protein_sequence(target_sequence)
        ligase_tokens = tokenize_protein_sequence(e3_ligase_sequence)

        target_sequence = self.esm.embed_sequence(target_sequence)
        e3_ligase_sequence = self.esm.embed_sequence(e3_ligase_sequence)

        sample = {
            "target_embed": target_sequence,
            "target_tokens": target_tokens,
            "warhead_graph": warhead,
            "linker_graph": linker,
            "e3_ligand_graph": ligase_ligand,
            "ligase_embed": e3_ligase_sequence,
            "ligase_tokens": ligase_tokens,
            "label": int(label),
        }

        if self.use_descriptors:
            warhead_smi = row.get('warhead_smiles', '')
            linker_smi = row.get('linker_smiles', '')
            e3_smi = row.get('e3_ligase_smiles', '')
            desc_w = compute_mol_descriptors(warhead_smi, self.fp_dim)
            desc_e = compute_mol_descriptors(e3_smi, self.fp_dim)
            if self.desc_mode == "whole":
                full_smi = f"{warhead_smi}.{linker_smi}.{e3_smi}"
                sample["mol_descriptors"] = torch.tensor(
                    compute_mol_descriptors(full_smi, self.fp_dim), dtype=torch.float32
                )
            elif self.desc_mode == "WaE3":
                sample["mol_descriptors"] = torch.tensor(
                    np.concatenate([desc_w, desc_e]), dtype=torch.float32
                )
            else:
                desc_l = compute_mol_descriptors(linker_smi, self.fp_dim)
                sample["mol_descriptors"] = torch.tensor(
                    np.concatenate([desc_w, desc_l, desc_e]), dtype=torch.float32
                )

        return sample

def collater(data_list):
    batch = {}
    target_embed = [x["target_embed"] for x in data_list]
    target_tokens = [x["target_tokens"] for x in data_list]
    warhead_graph = [x["warhead_graph"] for x in data_list]
    linker_graph = [x["linker_graph"] for x in data_list]
    e3_ligand_graph = [x["e3_ligand_graph"] for x in data_list]
    ligase_embed = [x["ligase_embed"] for x in data_list]
    ligase_tokens = [x["ligase_tokens"] for x in data_list]
    label = [x["label"] for x in data_list]

    batch["target_embed"] = pad_sequence(target_embed, batch_first=True)  # [B, N_t, 320]
    batch["target_tokens"] = pad_sequence(target_tokens, batch_first=True, padding_value=0)
    batch["warhead_graph"] = Batch.from_data_list(warhead_graph)
    batch["linker_graph"] = Batch.from_data_list(linker_graph)
    batch["e3_ligand_graph"] = Batch.from_data_list(e3_ligand_graph)
    batch["ligase_embed"] = pad_sequence(ligase_embed, batch_first=True)  # [B, N_l, 320]
    batch["ligase_tokens"] = pad_sequence(ligase_tokens, batch_first=True, padding_value=0)
    batch["label"] = torch.tensor(label)

    if "mol_descriptors" in data_list[0]:
        batch["mol_descriptors"] = torch.stack([x["mol_descriptors"] for x in data_list])

    return batch
