from typing import Optional, Tuple

from rdkit import Chem

# LoQI-supported element set for SMILES-based conformer generation.
SUPPORTED_ELEMENTS = {
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Si",
    "P",
    "S",
    "Cl",
    "As",
    "Br",
    "I",
    "Se",
}


def _check_mol_compatibility(mol_h: Chem.Mol) -> Optional[str]:
    for atom in mol_h.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in SUPPORTED_ELEMENTS:
            return (
                f"Unsupported element: '{symbol}'. Supported: "
                f"{', '.join(sorted(SUPPORTED_ELEMENTS))}."
            )
        if atom.GetNumRadicalElectrons() > 0:
            return (
                f"Radical electrons are not supported "
                f"(atom {atom.GetIdx()} {atom.GetSymbol()})."
            )
    return None


def validate_smiles(
        smiles: str,
        add_hs: bool = True,
) -> Tuple[Optional[Chem.Mol], Optional[str], Optional[str]]:
    """Validate and revalidate SMILES for LoQI inference."""
    if smiles is None or not str(smiles).strip():
        return None, None, "Empty SMILES string."

    smiles = str(smiles).strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, f"RDKit failed to parse SMILES: {smiles!r}."

    if len(Chem.GetMolFrags(mol)) > 1:
        return None, None, "Disconnected fragments are not supported."

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    mol_roundtrip = Chem.MolFromSmiles(canonical)
    if mol_roundtrip is None:
        return None, None, f"Revalidation failed after canonicalization: {canonical!r}."

    mol_checked = Chem.AddHs(mol_roundtrip) if add_hs else mol_roundtrip
    compatibility_error = _check_mol_compatibility(mol_checked)
    if compatibility_error is not None:
        return None, None, compatibility_error

    return mol_checked, canonical, None


def validate_rdkit_mol(mol: Chem.Mol, add_hs: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """Validate an RDKit molecule by SMILES roundtrip + LoQI compatibility checks."""
    if mol is None:
        return None, "Empty RDKit molecule."
    if mol.GetNumAtoms() == 0:
        return None, "RDKit molecule has zero atoms."

    try:
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception as exc:
        return None, f"Could not convert molecule to SMILES: {exc}"

    _, canonical, error = validate_smiles(smiles, add_hs=add_hs)
    if error is not None:
        return None, error
    return canonical, None
