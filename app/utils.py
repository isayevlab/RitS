"""
Utility functions for RitS transition-state sampling app.
"""

import sys
import tempfile
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf
from megalodon.models.module import Graph3DInterpolantModel
from megalodon.data.ts_batch_preprocessor import TsBatchPreProcessor
from megalodon.metrics.ts_evaluation_callback import convert_coords_to_np

Chem.SetUseLegacyStereoPerception(True)

BOND_TYPES = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}
NUM_BOND_TYPES = 9

CONFIG_PATH = ROOT / "scripts" / "conf" / "rits.yaml"
CKPT_PATH = ROOT / "data" / "rits.ckpt"

EXAMPLE_REACTIONS = {
    "Amide hydrolysis": (
        "[C:1]([N:2]([C:4]([C:3]([H:10])([H:11])[H:12])=[O:13])[H:8])([H:5])([H:6])[H:7]"
        ".[H:9][O:14][H:15]"
        ">>"
        "[C:1]([N:2]([H:8])[H:9])([H:5])([H:6])[H:7]"
        ".[C:3]([C:4](=[O:13])[O:14][H:15])([H:10])([H:11])[H:12]"
    ),
    "Diels-Alder": (
        "[C:12](#[C:13][C@@:14]1([H:29])[C:15]([H:30])([H:31])[N:16]([H:32])"
        "[C:17]([H:33])([H:34])[C:18]1([H:35])[H:36])[H:28]"
        ".[C:1](=[C:2](/[C:3](=[C:4](\\[C:5]([C:6](=[O:7])[O:8][C:9](=[O:10])"
        "[N:11]([H:26])[H:27])([H:24])[H:25])[H:23])[H:22])[H:21])([H:19])[H:20]"
        ">>"
        "[C:1]1([H:19])([H:20])[C:2]([H:21])=[C:3]([H:22])[C@:4]([C:5]([C:6](=[O:7])"
        "[O:8][C:9](=[O:10])[N:11]([H:26])[H:27])([H:24])[H:25])([H:23])"
        "[C:12]([H:28])=[C:13]1[C@@:14]1([H:29])[C:15]([H:30])([H:31])"
        "[N:16]([H:32])[C:17]([H:33])([H:34])[C:18]1([H:35])[H:36]"
    ),
    "Click chemistry (CuAAC)": (
        "[C:1]([C:2](=[O:3])[O:4][C:5]([C:6]([C:7]([N:8]=[N+:9]=[N-:10])"
        "([H:18])[H:19])([H:16])[H:17])([H:14])[H:15])([H:11])([H:12])[H:13]"
        ".[O:20]([C:21]([C:22]#[C:23][H:27])([H:25])[H:26])[H:24]"
        ">>"
        "[C@:1]([C:2](=[O:3])[O:4][C@@:5]([C@@:6]([C@@:7]([N:8]1:[N:9]:[N:10]:"
        "[C:22]([C@:21]([O:20][H:24])([H:25])[H:26]):[C:23]:1[H:27])"
        "([H:18])[H:19])([H:16])[H:17])([H:14])[H:15])([H:11])([H:12])[H:13]"
    ),
    "Epoxidation": (
        "[H:11][O:19][C:20](=[C:22]([H:21])[H:23])[C:24]([H:25])([H:26])[H:27]"
        ".[O:1]([C:2](=[O:3])[C:4]([c:5]1[c:6]([H:14])[c:7]([H:15])[c:8]([H:16])"
        "[c:9]([H:17])[c:10]1[H:18])([H:12])[H:13])[H:28]"
        ">>"
        "[O:19]1[C:20]([C:24]([H:25])([H:26])[H:27])([H:28])[C:22]1([H:21])[H:23]"
        ".[O:1]=[C:2]([O:3][H:11])[C:4]([c:5]1[c:6]([H:14])[c:7]([H:15])"
        "[c:8]([H:16])[c:9]([H:17])[c:10]1[H:18])([H:12])[H:13]"
    ),
    "Ester exchange": (
        "[C:1]([C:2]([O:3][C:4](=[O:5])[C:6]([C:7]([C:8]([H:17])([H:18])[H:19])"
        "=[O:9])([H:15])[H:16])([H:13])[H:14])([H:10])([H:11])[H:12]"
        ".[O:20]([C:23]([H:21])([H:22])[H:24])[H:25]"
        ">>"
        "[C:1]([C:2]([O:3][C:4](=[O:5])[C:6]([H:15])([H:16])[H:25])([H:13])"
        "[H:14])([H:10])([H:11])[H:12]"
        ".[C:7]([C:8]([H:17])([H:18])[H:19])(=[O:9])[O:20][C:23]([H:21])([H:22])[H:24]"
    ),
    "Carbamate formation": (
        "[C:11]([H:12])([H:13])([H:14])[N:15]=[C:16]=[O:17]"
        ".[C:1](=[C:2]([C:3]([O:4][H:5])([H:9])[H:10])[H:8])([H:6])[H:7]"
        ">>"
        "[C:1](=[C:2](/[C@@:3]([O:4][C:16]([N:15]([H:5])[C@@:11]([H:12])([H:13])"
        "[H:14])=[O:17])([H:9])[H:10])[H:8])(\\[H:6])[H:7]"
    ),
    "E2 elimination (chlorostyrene)": (
        "[C:1]1([C:7]([C:8]([Cl:9])([Cl:10])[H:18])([H:16])[H:17])=[C:2]([H:11])"
        "[C:3]([H:12])=[C:4]([H:13])[C:5]([H:14])=[C:6]1[H:15]"
        ">>"
        "[C:1]1([C:7](=[C:8]([Cl:9])[H:18])[H:16])=[C:2]([H:11])[C:3]([H:12])"
        "=[C:4]([H:13])[C:5]([H:14])=[C:6]1[H:15].[Cl:10][H:17]"
    ),
}

# ---------------------------------------------------------------------------
# Stereo bond helpers (from sample_transition_state.py)
# ---------------------------------------------------------------------------

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
            idx_5 = [n.GetIdx() for n in atom_1.GetNeighbors() if n.GetIdx() not in {idx_2, idx_3}]
            idx_6 = [n.GetIdx() for n in atom_2.GetNeighbors() if n.GetIdx() not in {idx_1, idx_4}]
            inv_stereo = (
                Chem.BondStereo.STEREOE if stereo == Chem.BondStereo.STEREOZ else Chem.BondStereo.STEREOZ
            )
            result.extend([(idx_3, idx_4, ez_bonds[stereo]), (idx_4, idx_3, ez_bonds[stereo])])
            if idx_5:
                result.extend([(idx_5[0], idx_4, ez_bonds[inv_stereo]),
                               (idx_4, idx_5[0], ez_bonds[inv_stereo])])
            if idx_6:
                result.extend([(idx_3, idx_6[0], ez_bonds[inv_stereo]),
                               (idx_6[0], idx_3, ez_bonds[inv_stereo])])
            if idx_5 and idx_6:
                result.extend([(idx_5[0], idx_6[0], ez_bonds[stereo]),
                               (idx_6[0], idx_5[0], ez_bonds[stereo])])

        if bond.GetBeginAtom().HasProp("_CIPCode"):
            chirality = bond.GetBeginAtom().GetProp("_CIPCode")
            neighbors = bond.GetBeginAtom().GetNeighbors()
            if all(n.HasProp("_CIPRank") for n in neighbors):
                sorted_nbrs = sorted(neighbors, key=lambda x: int(x.GetProp("_CIPRank")), reverse=True)
                sorted_nbrs = [a.GetIdx() for a in sorted_nbrs]
                a_idx, b_idx, c_idx = sorted_nbrs[:3] if chirality == "R" else sorted_nbrs[:3][::-1]
                d_idx = sorted_nbrs[-1]
                result.extend([
                    (a_idx, d_idx, chi_bonds[0]), (b_idx, d_idx, chi_bonds[0]),
                    (c_idx, d_idx, chi_bonds[0]), (d_idx, a_idx, chi_bonds[0]),
                    (d_idx, b_idx, chi_bonds[0]), (d_idx, c_idx, chi_bonds[0]),
                    (b_idx, a_idx, chi_bonds[1]), (c_idx, b_idx, chi_bonds[1]),
                    (a_idx, c_idx, chi_bonds[1]),
                ])

    for i, j, v in result:
        if bmat[i, j] == 0:
            bmat[i, j] = v
    return bmat


# ---------------------------------------------------------------------------
# Reaction SMARTS → PyG Data (from sample_transition_state.py)
# ---------------------------------------------------------------------------

def process_reaction_smarts(r_smarts, p_smarts, charge=0, add_stereo=True):
    """Parse mapped reactant/product SMILES into a PyG Data object.
    Always kekulizes (model trained on kekulized bonds).
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    r = Chem.MolFromSmiles(r_smarts, params)
    p = Chem.MolFromSmiles(p_smarts, params)

    Chem.Kekulize(r, clearAromaticFlags=True)
    Chem.Kekulize(p, clearAromaticFlags=True)

    N = r.GetNumAtoms()
    assert p.GetNumAtoms() == N

    r_perm = np.array([a.GetAtomMapNum() for a in r.GetAtoms()]) - 1
    p_perm = np.array([a.GetAtomMapNum() for a in p.GetAtoms()]) - 1
    r_perm_inv = np.argsort(r_perm)
    p_perm_inv = np.argsort(p_perm)

    r_atomic = np.array([r.GetAtomWithIdx(int(i)).GetAtomicNum() for i in r_perm_inv])
    p_atomic = np.array([p.GetAtomWithIdx(int(i)).GetAtomicNum() for i in p_perm_inv])
    assert np.array_equal(r_atomic, p_atomic)
    numbers = torch.from_numpy(r_atomic).to(torch.uint8)

    r_adj = Chem.rdmolops.GetAdjacencyMatrix(r)
    p_adj = Chem.rdmolops.GetAdjacencyMatrix(p)
    r_adj_perm = r_adj[r_perm_inv, :].T[r_perm_inv, :].T
    p_adj_perm = p_adj[p_perm_inv, :].T[p_perm_inv, :].T

    adj = r_adj_perm + p_adj_perm
    row, col = adj.nonzero()

    _nonbond = 0
    r_edge_type, p_edge_type = [], []
    for i, j in zip(r_perm_inv[row], r_perm_inv[col]):
        b = r.GetBondBetweenAtoms(int(i), int(j))
        r_edge_type.append(BOND_TYPES.get(b.GetBondType(), 1) if b else _nonbond)
    for i, j in zip(p_perm_inv[row], p_perm_inv[col]):
        b = p.GetBondBetweenAtoms(int(i), int(j))
        p_edge_type.append(BOND_TYPES.get(b.GetBondType(), 1) if b else _nonbond)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    r_edge_type = torch.tensor(r_edge_type, dtype=torch.uint8)
    p_edge_type = torch.tensor(p_edge_type, dtype=torch.uint8)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    r_edge_type = r_edge_type[perm]
    p_edge_type = p_edge_type[perm]

    if add_stereo:
        chi_bonds = (7, 8)
        ez_bonds = {Chem.BondStereo.STEREOE: 5, Chem.BondStereo.STEREOZ: 6}
        r_bmat = np.zeros((N, N), dtype=np.int64)
        p_bmat = np.zeros((N, N), dtype=np.int64)
        for idx, (i, j) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
            r_bmat[i, j] = r_edge_type[idx].item()
            p_bmat[i, j] = p_edge_type[idx].item()

        r_reordered = Chem.RenumberAtoms(r, r_perm_inv.tolist())
        p_reordered = Chem.RenumberAtoms(p, p_perm_inv.tolist())
        r_bmat = add_stereo_bonds(r_reordered, chi_bonds, ez_bonds, r_bmat, from_3D=False)
        p_bmat = add_stereo_bonds(p_reordered, chi_bonds, ez_bonds, p_bmat, from_3D=False)

        existing = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        new_edges, new_r, new_p = [], [], []
        for i in range(N):
            for j in range(N):
                if i != j and (i, j) not in existing:
                    if r_bmat[i, j] > 0 or p_bmat[i, j] > 0:
                        new_edges.append((i, j))
                        new_r.append(r_bmat[i, j])
                        new_p.append(p_bmat[i, j])
        if new_edges:
            new_ei = torch.tensor(new_edges, dtype=torch.long).T
            edge_index = torch.cat([edge_index, new_ei], dim=1)
            r_edge_type = torch.cat([r_edge_type, torch.tensor(new_r, dtype=torch.uint8)])
            p_edge_type = torch.cat([p_edge_type, torch.tensor(new_p, dtype=torch.uint8)])
            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            r_edge_type = r_edge_type[perm]
            p_edge_type = p_edge_type[perm]

    edge_attr = torch.stack([r_edge_type, p_edge_type], dim=-1)
    pos = torch.zeros(N, 3, dtype=torch.float32)
    smiles = f"{r_smarts}>>{p_smarts}"

    return Data(
        numbers=numbers,
        charges=torch.full((N,), charge, dtype=torch.int8),
        ts_coord=pos,
        r_coord=pos.clone(),
        p_coord=pos.clone(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=N,
        id=smiles,
    )


# ---------------------------------------------------------------------------
# Common-bond topology (union of R and P adjacency, keeping only shared bonds)
# ---------------------------------------------------------------------------

def get_bond_topology_from_smarts(reaction_smarts: str):
    """Return (atomic_numbers, common_bonds) from a reaction SMARTS string.

    common_bonds is a list of (i, j) pairs (mapped-atom indices, 0-based)
    representing bonds that exist in BOTH reactant and product. This is used
    to display the TS 3D structure with a reasonable connectivity.
    """
    r_smarts, p_smarts = reaction_smarts.split(">>")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    r = Chem.MolFromSmiles(r_smarts, params)
    p = Chem.MolFromSmiles(p_smarts, params)

    N = r.GetNumAtoms()
    r_perm_inv = np.argsort(np.array([a.GetAtomMapNum() for a in r.GetAtoms()]) - 1)
    p_perm_inv = np.argsort(np.array([a.GetAtomMapNum() for a in p.GetAtoms()]) - 1)

    r_adj = Chem.rdmolops.GetAdjacencyMatrix(r)
    p_adj = Chem.rdmolops.GetAdjacencyMatrix(p)
    r_adj_perm = r_adj[r_perm_inv, :].T[r_perm_inv, :].T
    p_adj_perm = p_adj[p_perm_inv, :].T[p_perm_inv, :].T

    common = (r_adj_perm > 0) & (p_adj_perm > 0)
    bonds = []
    for i in range(N):
        for j in range(i + 1, N):
            if common[i, j]:
                bonds.append((i, j))

    atomic_numbers = [r.GetAtomWithIdx(int(idx)).GetAtomicNum() for idx in r_perm_inv]
    return atomic_numbers, bonds


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_ts_model(device: str = "auto"):
    """Load the RitS model (single config/checkpoint)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = OmegaConf.load(str(CONFIG_PATH))
    bp = TsBatchPreProcessor(
        aug_rotations=cfg.data.get("aug_rotations", False),
        scale_coords=cfg.data.get("scale_coords", 1.0),
    )
    model = Graph3DInterpolantModel.load_from_checkpoint(
        str(CKPT_PATH),
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=bp,
        strict=True,
    )
    return model.to(device).eval(), cfg, device


# ---------------------------------------------------------------------------
# TS sampling
# ---------------------------------------------------------------------------

def sample_transition_states(
    reaction_smarts: str,
    model,
    cfg,
    device: str,
    n_samples: int = 5,
    num_steps: int = 25,
    charge: int = 0,
    add_stereo: bool = True,
):
    """Sample TS geometries for a reaction. Returns list of (symbols, coords) tuples."""
    r_smi, p_smi = reaction_smarts.split(">>")
    data = process_reaction_smarts(r_smi, p_smi, charge=charge, add_stereo=add_stereo)

    all_data = [deepcopy(data) for _ in range(n_samples)]
    loader = DataLoader(all_data, batch_size=min(n_samples, 32))

    timesteps = num_steps if num_steps is not None else cfg.interpolant.timesteps
    results = []

    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            sample = model.sample(batch=batch, timesteps=timesteps, pre_format=True)
        coords_list = convert_coords_to_np(sample)
        for k in range(len(coords_list)):
            mask = batch["batch"] == k
            nums = batch["numbers"][mask].cpu().numpy()
            symbols = [Chem.GetPeriodicTable().GetElementSymbol(int(z)) for z in nums]
            results.append((symbols, coords_list[k]))

    return results


def coords_to_xyz_string(symbols, coords):
    """Convert (symbols, coords) to an XYZ-format string."""
    n = len(symbols)
    lines = [str(n), ""]
    for sym, c in zip(symbols, coords):
        lines.append(f"{sym} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
    return "\n".join(lines)


def multi_xyz_string(samples):
    """Concatenate multiple (symbols, coords) into a multi-frame XYZ string."""
    return "\n".join(coords_to_xyz_string(s, c) for s, c in samples) + "\n"


# ---------------------------------------------------------------------------
# 2D reaction visualisation (from reactions/visualize_reactions.py)
# ---------------------------------------------------------------------------

def _remove_h_heavy_only(mol):
    out = Chem.RWMol()
    heavy_idx = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        a = Chem.Atom(atom.GetAtomicNum())
        a.SetChiralTag(atom.GetChiralTag())
        if atom.GetFormalCharge() != 0:
            a.SetFormalCharge(atom.GetFormalCharge())
        heavy_idx[atom.GetIdx()] = out.AddAtom(a)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in heavy_idx and j in heavy_idx:
            out.AddBond(heavy_idx[i], heavy_idx[j], bond.GetBondType())
    return Chem.Mol(out)


def _strip_stereo(mol):
    mol = Chem.RWMol(mol)
    for bond in mol.GetBonds():
        bond.SetBondDir(Chem.BondDir.NONE)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    for atom in mol.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return Chem.Mol(mol)


def _clear_atom_maps(mol):
    mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.Mol(mol)


def render_reaction_svg(smarts: str, width: int = 1200, height: int = 360) -> Optional[str]:
    """Return an SVG string of the 2D reaction scheme (heavy atoms only, no stereo)."""
    try:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
    except Exception:
        rxn = None

    if rxn is None:
        parts = smarts.split(">>", 1)
        if len(parts) != 2:
            return None
        ps = Chem.SmilesParserParams()
        ps.removeHs = False
        rxn = rdChemReactions.ChemicalReaction()
        for frag in parts[0].split("."):
            mol = Chem.MolFromSmiles(frag.strip(), ps)
            if mol is None:
                return None
            rxn.AddReactantTemplate(mol)
        for frag in parts[1].split("."):
            mol = Chem.MolFromSmiles(frag.strip(), ps)
            if mol is None:
                return None
            rxn.AddProductTemplate(mol)
        rxn.Initialize()

    draw_rxn = rdChemReactions.ChemicalReaction()
    for i in range(rxn.GetNumReactantTemplates()):
        mol = Chem.Mol(rxn.GetReactantTemplate(i))
        mol = _remove_h_heavy_only(mol)
        mol = _clear_atom_maps(mol)
        mol = _strip_stereo(mol)
        try:
            rdDepictor.Compute2DCoords(mol)
        except Exception:
            pass
        draw_rxn.AddReactantTemplate(mol)
    for i in range(rxn.GetNumProductTemplates()):
        mol = Chem.Mol(rxn.GetProductTemplate(i))
        mol = _remove_h_heavy_only(mol)
        mol = _clear_atom_maps(mol)
        mol = _strip_stereo(mol)
        try:
            rdDepictor.Compute2DCoords(mol)
        except Exception:
            pass
        draw_rxn.AddProductTemplate(mol)
    draw_rxn.Initialize()

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 4.0
    opts.baseFontSize = 0.9
    opts.minFontSize = 14
    opts.maxFontSize = 36
    opts.padding = 0.05
    opts.addStereoAnnotation = False
    opts.drawMolsSameScale = True
    opts.clearBackground = True
    drawer.DrawReaction(draw_rxn, highlightByReactant=False)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# ---------------------------------------------------------------------------
# IRC helpers (from scripts/run_irc.py)
# ---------------------------------------------------------------------------

def parse_multiframe_xyz(text: str) -> list[tuple[int, str, list[str]]]:
    """Parse multi-frame XYZ text. Returns list of (n_atoms, comment, atom_lines)."""
    lines = text.strip().splitlines()
    frames = []
    i = 0
    while i < len(lines):
        try:
            n = int(lines[i].strip())
        except ValueError:
            i += 1
            continue
        i += 1
        if i >= len(lines):
            break
        comment = lines[i].strip()
        i += 1
        atom_lines = []
        for _ in range(n):
            if i >= len(lines):
                break
            atom_lines.append(lines[i])
            i += 1
        if len(atom_lines) == n:
            frames.append((n, comment, atom_lines))
    return frames


def frame_to_xyz_string(n, comment, atom_lines):
    return f"{n}\n{comment}\n" + "\n".join(atom_lines)


IRC_THRESH_OPTIONS = ["gau_loose", "gau", "gau_tight", "gau_vtight"]
IRC_TYPE_OPTIONS = ["eulerpc", "dvv", "gonzalez_schlegel", "lqa", "damped_velocity_verlet"]
TSOPT_TYPE_OPTIONS = ["rsprfo", "rsirfo", "trim"]


def write_pysis_yml(
    yml_path: Path,
    charge: int = 0,
    mult: int = 1,
    thresh: str = "gau",
    max_cycles: int = 1000,
    hessian_recalc: int = 5,
    trust_max: float = 0.05,
    trust_min: float = 0.001,
    tsopt_type: str = "rsprfo",
    irc_type: str = "eulerpc",
    pal: int = 1,
):
    """Write pysisyphus YAML for TS optimisation + IRC."""
    yml_path.write_text(f"""tsopt:
  type: {tsopt_type}
  hessian_recalc: {hessian_recalc}
  assert_neg_eigval: True
  trust_max: {trust_max}
  trust_min: {trust_min}
  thresh: {thresh}
  max_cycles: {max_cycles}
irc:
  type: {irc_type}
calc:
  type: xtb
  pal: {pal}
  charge: {charge}
  mult: {mult}
geom:
  type: cart
  fn: [ts.xyz]
""", encoding="utf-8")


def run_irc_for_xyz(
    xyz_string: str,
    charge: int = 0,
    mult: int = 1,
    thresh: str = "gau",
    max_cycles: int = 1000,
    hessian_recalc: int = 5,
    trust_max: float = 0.05,
    trust_min: float = 0.001,
    tsopt_type: str = "rsprfo",
    irc_type: str = "eulerpc",
    pal: int = 1,
):
    """Run pysisyphus IRC on a single TS XYZ string. Returns (success, irc_trj_text, run_dir)."""
    tmp = tempfile.mkdtemp(prefix="rits_irc_")
    run_dir = Path(tmp)
    ts_path = run_dir / "ts.xyz"
    ts_path.write_text(xyz_string + "\n", encoding="utf-8")
    write_pysis_yml(
        run_dir / "pysis.yml",
        charge=charge, mult=mult, thresh=thresh,
        max_cycles=max_cycles, hessian_recalc=hessian_recalc,
        trust_max=trust_max, trust_min=trust_min,
        tsopt_type=tsopt_type, irc_type=irc_type, pal=pal,
    )

    try:
        subprocess.run(
            ["pysis", "pysis.yml"],
            cwd=run_dir,
            check=True,
            capture_output=True,
            timeout=600,
        )
    except FileNotFoundError:
        return False, "pysisyphus (pysis) not found. Install with: pip install pysisyphus", run_dir
    except subprocess.CalledProcessError as e:
        return False, f"pysis failed: {e.stderr.decode()[-500:]}", run_dir
    except subprocess.TimeoutExpired:
        return False, "IRC calculation timed out (10 min limit)", run_dir

    trj_path = run_dir / "finished_irc.trj"
    if trj_path.is_file():
        return True, trj_path.read_text(encoding="utf-8"), run_dir
    return False, "IRC finished but no trajectory file produced", run_dir
