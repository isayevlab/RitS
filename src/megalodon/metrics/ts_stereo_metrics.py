"""
Stereochemistry evaluation metrics for transition states.

Key insight: TS geometry already contains stereochemical information of both 
reactants and products. We can evaluate stereo correctness by:
1. Overlaying TS coordinates with reactant/product topology
2. Computing stereochemistry from 3D coordinates
3. Comparing with ground truth from SMILES
"""
from typing import Dict, List, Tuple
import numpy as np
from rdkit import Chem

from megalodon.metrics.preserved_stereo import get_stereochemistry_descriptor


def create_mol_from_coords_and_bmat(coords: np.ndarray, 
                                     numbers: np.ndarray, 
                                     bmat: np.ndarray,
                                     charge: int = 0) -> Chem.Mol:
    """
    Create RDKit molecule from coordinates, atomic numbers, and bond matrix.
    
    Args:
        coords: Atomic coordinates (N, 3)
        numbers: Atomic numbers (N,)
        bmat: Bond matrix (N, N) with bond types (0=none, 1=single, 2=double, etc.)
        charge: Molecular charge
        
    Returns:
        RDKit Mol object with 3D coordinates
    """
    # Create editable molecule
    mol = Chem.RWMol()
    
    # Add atoms - numbers are always atomic numbers for TS dataset
    for atom_num in numbers:
        atomic_num = int(atom_num)
        atom = Chem.Atom(atomic_num)
        mol.AddAtom(atom)
    
    # Add bonds (only upper triangle to avoid duplicates)
    bond_type_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    
    n_atoms = len(numbers)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            bond_order = int(bmat[i, j])
            # Only consider standard bond types (1-4), ignore stereo encoding (5-8)
            if 1 <= bond_order <= 4:
                mol.AddBond(i, j, bond_type_map[bond_order])
    
    # Convert to regular mol
    mol = mol.GetMol()
    
    # Add 3D coordinates
    conf = Chem.Conformer(n_atoms)
    for i in range(n_atoms):
        conf.SetAtomPosition(i, coords[i].tolist())
    mol.AddConformer(conf)
    
    # Set charge
    mol.SetProp("_TotalCharge", str(charge))
    
    return mol


class TSStereoMetrics:
    """
    Compute stereochemistry accuracy for transition states.
    
    Evaluates whether generated TS coordinates preserve the stereochemistry
    encoded in reactant and product topologies.
    """
    
    def __init__(self):
        self.total_r_rs = 0
        self.correct_r_rs = 0
        self.total_r_ez = 0
        self.correct_r_ez = 0
        
        self.total_p_rs = 0
        self.correct_p_rs = 0
        self.total_p_ez = 0
        self.correct_p_ez = 0
    
    def reset(self):
        """Reset all counters."""
        self.total_r_rs = 0
        self.correct_r_rs = 0
        self.total_r_ez = 0
        self.correct_r_ez = 0
        
        self.total_p_rs = 0
        self.correct_p_rs = 0
        self.total_p_ez = 0
        self.correct_p_ez = 0
    
    def evaluate_single(self, 
                       ts_coords: np.ndarray,
                       r_coords: np.ndarray, 
                       p_coords: np.ndarray,
                       numbers: np.ndarray,
                       bmat_r: np.ndarray,
                       bmat_p: np.ndarray,
                       charge: int = 0) -> Dict[str, int]:
        """
        Evaluate stereochemistry for a single TS.
        
        Args:
            ts_coords: Generated TS coordinates (N, 3)
            r_coords: Ground truth reactant coordinates (N, 3) 
            p_coords: Ground truth product coordinates (N, 3)
            numbers: Atomic numbers (N,)
            bmat_r: Reactant bond matrix (N, N)
            bmat_p: Product bond matrix (N, N)
            charge: Molecular charge
            
        Returns:
            Dict with counts of correct/total stereo for reactant and product
        """
        results = {
            'r_rs_correct': 0, 'r_rs_total': 0,
            'r_ez_correct': 0, 'r_ez_total': 0,
            'p_rs_correct': 0, 'p_rs_total': 0,
            'p_ez_correct': 0, 'p_ez_total': 0,
        }
        
        try:
            # Create molecules with TS coords + reactant/product topology
            ts_as_reactant = create_mol_from_coords_and_bmat(
                ts_coords, numbers, bmat_r, charge
            )
            ts_as_product = create_mol_from_coords_and_bmat(
                ts_coords, numbers, bmat_p, charge
            )
            
            # Create reference molecules with ground truth coords
            ref_reactant = create_mol_from_coords_and_bmat(
                r_coords, numbers, bmat_r, charge
            )
            ref_product = create_mol_from_coords_and_bmat(
                p_coords, numbers, bmat_p, charge
            )
            
            # Assign stereochemistry from 3D
            Chem.rdmolops.AssignStereochemistryFrom3D(ts_as_reactant)
            Chem.rdmolops.AssignStereochemistryFrom3D(ts_as_product)
            Chem.rdmolops.AssignStereochemistryFrom3D(ref_reactant)
            Chem.rdmolops.AssignStereochemistryFrom3D(ref_product)
            
            # Get descriptors for reactant
            r_sr, r_inv_sr, r_ez = get_stereochemistry_descriptor(ts_as_reactant)
            ref_r_sr, _, ref_r_ez = get_stereochemistry_descriptor(ref_reactant)
            
            # Check reactant R/S (absolute stereochemistry)
            if ref_r_sr:
                results['r_rs_total'] = 1
                if r_sr == ref_r_sr:
                    results['r_rs_correct'] = 1
            
            # Check reactant E/Z
            if ref_r_ez:
                results['r_ez_total'] = 1
                if r_ez == ref_r_ez:
                    results['r_ez_correct'] = 1
            
            # Get descriptors for product
            p_sr, p_inv_sr, p_ez = get_stereochemistry_descriptor(ts_as_product)
            ref_p_sr, _, ref_p_ez = get_stereochemistry_descriptor(ref_product)
            
            # Check product R/S (absolute stereochemistry)
            if ref_p_sr:
                results['p_rs_total'] = 1
                if p_sr == ref_p_sr:
                    results['p_rs_correct'] = 1
            
            # Check product E/Z
            if ref_p_ez:
                results['p_ez_total'] = 1
                if p_ez == ref_p_ez:
                    results['p_ez_correct'] = 1
                    
        except Exception as e:
            # Skip molecules that fail (e.g., invalid chemistry)
            print(f"Warning: Failed to evaluate TS stereo: {e}")
        
        return results
    
    def __call__(self, molecules_data: List[Dict]) -> Dict[str, float]:
        """
        Compute stereochemistry accuracy for a batch of TS molecules.
        
        Args:
            molecules_data: List of dicts with keys:
                - coords: TS coordinates (N, 3)
                - r_coords: Reactant coordinates (N, 3)
                - p_coords: Product coordinates (N, 3)
                - numbers: Atomic numbers (N,)
                - bmat_r: Reactant bond matrix (N, N)
                - bmat_p: Product bond matrix (N, N)
                - charge: Molecular charge
                
        Returns:
            Dict with stereo accuracy metrics
        """
        self.reset()
        
        for mol_data in molecules_data:
            results = self.evaluate_single(
                mol_data['coords'],
                mol_data['r_coords'],
                mol_data['p_coords'],
                mol_data['numbers'],
                mol_data['bmat_r'],
                mol_data['bmat_p'],
                mol_data.get('charge', 0)
            )
            
            # Accumulate counts
            self.total_r_rs += results['r_rs_total']
            self.correct_r_rs += results['r_rs_correct']
            self.total_r_ez += results['r_ez_total']
            self.correct_r_ez += results['r_ez_correct']
            
            self.total_p_rs += results['p_rs_total']
            self.correct_p_rs += results['p_rs_correct']
            self.total_p_ez += results['p_ez_total']
            self.correct_p_ez += results['p_ez_correct']
        
        # Compute accuracy scores
        metrics = {}
        
        # Reactant metrics
        metrics['ts_reactant_rs_score'] = (
            self.correct_r_rs / self.total_r_rs if self.total_r_rs > 0 else 1.0
        )
        metrics['ts_reactant_ez_score'] = (
            self.correct_r_ez / self.total_r_ez if self.total_r_ez > 0 else 1.0
        )
        
        # Product metrics
        metrics['ts_product_rs_score'] = (
            self.correct_p_rs / self.total_p_rs if self.total_p_rs > 0 else 1.0
        )
        metrics['ts_product_ez_score'] = (
            self.correct_p_ez / self.total_p_ez if self.total_p_ez > 0 else 1.0
        )
        
        # Combined metrics
        total_rs = self.total_r_rs + self.total_p_rs
        correct_rs = self.correct_r_rs + self.correct_p_rs
        metrics['ts_overall_rs_score'] = (
            correct_rs / total_rs if total_rs > 0 else 1.0
        )
        
        total_ez = self.total_r_ez + self.total_p_ez
        correct_ez = self.correct_r_ez + self.correct_p_ez
        metrics['ts_overall_ez_score'] = (
            correct_ez / total_ez if total_ez > 0 else 1.0
        )
        
        # Add counts for debugging
        metrics['ts_n_reactant_rs'] = self.total_r_rs
        metrics['ts_n_reactant_ez'] = self.total_r_ez
        metrics['ts_n_product_rs'] = self.total_p_rs
        metrics['ts_n_product_ez'] = self.total_p_ez
        
        return metrics

