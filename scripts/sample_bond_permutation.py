#!/usr/bin/env python3
"""
Simplified sampling script for bond permutation pipeline.
Reads a single molecule (SMILES or XYZ), creates bond matrix with kekulize and stereo,
then uses permute(bmat_p, mol) to get bmat_r.
"""
import os
import tempfile
import subprocess
from copy import deepcopy
from argparse import ArgumentParser

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Geometry import Point3D

from tqdm import tqdm
from torch_geometric.data import DataLoader
import torch
import numpy as np
from omegaconf import OmegaConf
from torch_geometric.data import Data

from megalodon.models.module import Graph3DInterpolantModel
from megalodon.data.ts_batch_preprocessor import TsBatchPreProcessor
from megalodon.metrics.ts_evaluation_callback import convert_coords_to_np

# Simple bond type encoding:
# 0: no bond, 1: single, 2: double, 3: triple, 4: aromatic, 5-8: stereo bonds
BOND_TYPES = {
    BT.SINGLE: 1,
    BT.DOUBLE: 2,
    BT.TRIPLE: 3,
    BT.AROMATIC: 4,
}

Chem.SetUseLegacyStereoPerception(True)


def infer_bonds_with_obabel(xyz_path, charge=0):
    """Infer bonds using Open Babel CLI."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mol', delete=False) as f:
        mol_file = f.name

    cmd = ["obabel", xyz_path, "-O", mol_file, "-c", "--quiet", str(charge)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=30)

    mol = Chem.MolFromMolFile(mol_file, sanitize=False, removeHs=False)
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    os.unlink(mol_file)

    return mol


def build_rdkit_mol(numbers, coords, bond_mat):
    """Build RDKit molecule from atomic numbers, coordinates, and bond matrix."""
    mol = Chem.RWMol()
    bond_num_to_type = {v: k for k, v in BOND_TYPES.items()}
    for num in numbers:
        atom = Chem.Atom(int(num))
        mol.AddAtom(atom)

    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if bond_mat[i, j] in bond_num_to_type:
                bond_type = bond_num_to_type[bond_mat[i, j]]
                mol.AddBond(i, j, bond_type)

    mol = mol.GetMol()
    conf = Chem.Conformer(len(numbers))
    for i, pos in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(*pos))
    mol.AddConformer(conf, assignId=True)
    return mol


def add_stereo_bonds(mol, chi_bonds, ez_bonds, bmat, from_3D=False):
    """Add stereo bond information to adjacency matrix."""
    result = []
    if from_3D:
        Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    else:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if bond.GetBondType() == Chem.BondType.DOUBLE and stereo in ez_bonds:
            idx_3, idx_4 = bond.GetStereoAtoms()
            atom_1, atom_2 = bond.GetBeginAtom(), bond.GetEndAtom()
            idx_1, idx_2 = atom_1.GetIdx(), atom_2.GetIdx()

            idx_5 = [nbr.GetIdx() for nbr in atom_1.GetNeighbors() if nbr.GetIdx() not in {idx_2, idx_3}]
            idx_6 = [nbr.GetIdx() for nbr in atom_2.GetNeighbors() if nbr.GetIdx() not in {idx_1, idx_4}]

            inv_stereo = Chem.BondStereo.STEREOE if stereo == Chem.BondStereo.STEREOZ else Chem.BondStereo.STEREOZ
            result.extend([(idx_3, idx_4, ez_bonds[stereo]), (idx_4, idx_3, ez_bonds[stereo])])

            if idx_5:
                result.extend([(idx_5[0], idx_4, ez_bonds[inv_stereo]), (idx_4, idx_5[0], ez_bonds[inv_stereo])])
            if idx_6:
                result.extend([(idx_3, idx_6[0], ez_bonds[inv_stereo]), (idx_6[0], idx_3, ez_bonds[inv_stereo])])
            if idx_5 and idx_6:
                result.extend([(idx_5[0], idx_6[0], ez_bonds[stereo]), (idx_6[0], idx_5[0], ez_bonds[stereo])])

        if bond.GetBeginAtom().HasProp('_CIPCode'):
            chirality = bond.GetBeginAtom().GetProp('_CIPCode')
            neighbors = bond.GetBeginAtom().GetNeighbors()
            if all(n.HasProp("_CIPRank") for n in neighbors):
                sorted_neighbors = sorted(neighbors, key=lambda x: int(x.GetProp("_CIPRank")), reverse=True)
                sorted_neighbors = [a.GetIdx() for a in sorted_neighbors]
                a_idx, b_idx, c_idx = sorted_neighbors[:3] if chirality == "R" else sorted_neighbors[:3][::-1]
                d_idx = sorted_neighbors[-1]
                result.extend([
                    (a_idx, d_idx, chi_bonds[0]), (b_idx, d_idx, chi_bonds[0]), (c_idx, d_idx, chi_bonds[0]),
                    (d_idx, a_idx, chi_bonds[0]), (d_idx, b_idx, chi_bonds[0]), (d_idx, c_idx, chi_bonds[0]),
                    (b_idx, a_idx, chi_bonds[1]), (c_idx, b_idx, chi_bonds[1]), (a_idx, c_idx, chi_bonds[1])
                ])

    if len(result) > 0:
        for i, j, v in result:
            if bmat[i, j] == 0:
                bmat[i, j] = v
    return bmat


def load_molecule(input_path_or_str, charge=0):
    """Load a single molecule from SMILES string or XYZ file."""
    if os.path.isfile(input_path_or_str):
        if input_path_or_str.endswith(".xyz"):
            mol = infer_bonds_with_obabel(input_path_or_str, charge=charge)
        else:
            raise ValueError(f"Unsupported file format: {input_path_or_str}. Use .xyz file.")
    else:
        # Treat as SMILES string
        parser_params = Chem.SmilesParserParams()
        parser_params.removeHs = False
        parser_params.sanitize = False
        mol = Chem.MolFromSmiles(input_path_or_str, parser_params)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES: {input_path_or_str}")
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)

        # Generate 3D coordinates
        embed_params = AllChem.ETKDGv3()
        embed_params.randomSeed = 42
        embed_params.useRandomCoords = True
        AllChem.EmbedMolecule(mol, embed_params)

    return mol


def permute(bmat_p, mol):
    """
    Permute bond matrix to create reactant bond matrix.
    This is a placeholder - implement your permutation logic here.

    Args:
        bmat_p: Product bond matrix (N x N numpy array)
        mol: RDKit molecule

    Returns:
        bmat_r: Reactant bond matrix (N x N numpy array)
    """
    # TODO: Implement your bond permutation logic here
    # For now, just return a copy of bmat_p
    return bmat_p.copy()


def create_permutation_data(mol, charge=0):
    """
    Create a Data object for bond permutation sampling.
    Uses permute(bmat_p, mol) to get bmat_r.
    """
    # Get atomic numbers
    numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.uint8)
    N = len(numbers)

    # Get coordinates
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule must have 3D coordinates.")
    coords = mol.GetConformer().GetPositions()

    # Get product bond matrix
    bond_mat = Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True)
    bond_mat[bond_mat == 1.5] = 4  # aromatic
    bond_mat = bond_mat.astype(np.int32)

    # Build RDKit molecule for stereo assignment
    mol_for_stereo = build_rdkit_mol(numbers, coords, bond_mat)

    # Add stereo bonds
    chi_bonds = [7, 8]
    ez_bonds = {
        Chem.BondStereo.STEREOE: 5,
        Chem.BondStereo.STEREOZ: 6
    }
    bmat_p = add_stereo_bonds(mol_for_stereo, chi_bonds, ez_bonds, bond_mat.copy(), from_3D=True)

    # Get reactant bond matrix via permutation
    bmat_r = permute(bmat_p, mol)

    # Convert to torch tensors
    numbers_t = torch.from_numpy(numbers).to(torch.uint8)
    coord = torch.from_numpy(coords).float()
    bmat_r_t = torch.from_numpy(bmat_r)
    bmat_p_t = torch.from_numpy(bmat_p)
    charges = torch.full_like(numbers_t, charge, dtype=torch.int8)

    # Create edge index and edge attributes
    edge_index = (bmat_r_t + bmat_p_t).nonzero().contiguous().T
    edge_attr = torch.stack([bmat_r_t, bmat_p_t], dim=-1)[edge_index[0], edge_index[1]].to(torch.uint8)

    # Use same coordinates for r_coord, p_coord, ts_coord (will be replaced by model)
    return Data(
        numbers=numbers_t,
        charges=charges,
        r_coord=coord.clone(),
        p_coord=coord.clone(),
        ts_coord=coord.clone(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=N,
        id=f"permute_{N}",
    )


def coords_to_xyz_string(coords, numbers):
    """Convert coordinates and atomic numbers to XYZ format string."""
    n_atoms = len(numbers)
    xyz_lines = [str(n_atoms), ""]

    for atomic_num, coord in zip(numbers, coords):
        symbol = Chem.GetPeriodicTable().GetElementSymbol(int(atomic_num))
        xyz_lines.append(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")

    return "\n".join(xyz_lines)


def main():
    parser = ArgumentParser(description="Sample with bond permutation pipeline")

    # Input options
    parser.add_argument("--input", type=str, required=True,
                        help="Input molecule (SMILES string or XYZ file path)")

    # Model options
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint file")
    parser.add_argument("--output", type=str, required=True, help="Output XYZ file path")

    # Sampling options
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--charge", type=int, default=0, help="Molecular charge")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Number of diffusion steps (overrides config)")

    args = parser.parse_args()

    # Load molecule
    print(f"Loading molecule from: {args.input}")
    mol = load_molecule(args.input, charge=args.charge)
    print(f"Loaded molecule with {mol.GetNumAtoms()} atoms")

    # Load model
    cfg = OmegaConf.load(args.config)
    batch_preprocessor = TsBatchPreProcessor(
        aug_rotations=cfg.data.get("aug_rotations", False),
        scale_coords=cfg.data.get("scale_coords", 1.0),
    )

    model = Graph3DInterpolantModel.load_from_checkpoint(
        args.ckpt,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=batch_preprocessor,
        strict=True,
    )
    model = model.to("cuda").eval()

    # Create data
    data = create_permutation_data(mol, charge=args.charge)

    # Replicate for n_samples
    all_data_list = [deepcopy(data) for _ in range(args.n_samples)]
    print(f"Created {len(all_data_list)} samples")

    loader = DataLoader(all_data_list, batch_size=args.batch_size)

    # Sampling
    generated_coords = []
    reference_data = []

    timesteps = args.num_steps if args.num_steps is not None else cfg.interpolant.timesteps

    for batch in tqdm(loader, desc="Sampling"):
        batch = batch.to(model.device)

        with torch.no_grad():
            sample = model.sample(batch=batch, timesteps=timesteps, pre_format=True)

        coords_list = convert_coords_to_np(sample)
        generated_coords.extend(coords_list)

        for i in range(len(coords_list)):
            batch_mask = batch["batch"] == i
            ref_data = {
                "numbers": batch["numbers"][batch_mask].cpu().numpy(),
                "id": batch["id"][i] if isinstance(batch["id"][i], str) else str(batch["id"][i]),
            }
            reference_data.append(ref_data)

    # Save output
    if len(generated_coords) == 1:
        xyz_content = coords_to_xyz_string(generated_coords[0], reference_data[0]["numbers"])
        with open(args.output, "w") as f:
            f.write(xyz_content)
    else:
        with open(args.output, "w") as f:
            for coords, ref_data in zip(generated_coords, reference_data):
                xyz_content = coords_to_xyz_string(coords, ref_data["numbers"])
                f.write(xyz_content + "\n")

    print(f"Generated {len(generated_coords)} samples.")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
