from os.path import join
from pathlib import Path

import rdkit
import rdkit.Chem as Chem
from matplotlib import colors
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import MolToImage


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


def find_matches_one(mol, submol):
    # find all matching atoms for each submol in submol_list in mol.
    match_dict = {}
    mols = [mol, submol]  # pairwise search
    res = rdFMCS.FindMCS(mols)  # ringMatchesRingOnly=True)
    mcsp = Chem.MolFromSmarts(res.smartsString)
    matches = mol.GetSubstructMatches(mcsp)
    return matches


# Draw the molecule
def get_image(mol, output_dir, file_name, atomset=None):
    hcolor = colors.to_rgb("green")
    if atomset is not None:
        # highlight the atoms set while drawing the whole molecule.
        img = MolToImage(
            mol,
            size=(600, 600),
            fitImage=True,
            highlightAtoms=atomset,
            highlightColor=hcolor,
        )
    else:
        img = MolToImage(mol, size=(400, 400), fitImage=True)
    img.save(join(Path(output_dir).absolute(), file_name))
    return img
