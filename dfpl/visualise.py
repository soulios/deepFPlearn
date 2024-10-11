import rdkit.Chem as Chem
from matplotlib import colors
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import MolToImage
import math
from PIL import Image


def get_mol(smiles):
    if isinstance(smiles, list):
        smiles = smiles[0]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


def find_matches_one(mol, submol):
    # find all matching atoms for each submol in submol_list in mol.
    mols = [mol, submol]  # pairwise search
    res = rdFMCS.FindMCS(mols)  # ringMatchesRingOnly=True)
    mcsp = Chem.MolFromSmarts(res.smartsString)
    matches = mol.GetSubstructMatches(mcsp)
    return matches


# Draw the molecule
def get_image(mol, atomset=None):
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
    return img


def calculate_grid_size(total_images):
    """
    Calculate the grid size that best fits the total number of images.
    The goal is to make the grid as square as possible.

    :param total_images: Total number of images to display.
    :return: (rows, cols) that fit the total number of images.
    """
    cols = math.ceil(math.sqrt(total_images))  # Number of columns
    rows = math.ceil(total_images / cols)  # Number of rows

    return (rows, cols)


def combine_images(image_list, grid_size, image_size=(400, 400)):
    """
    Combines individual images into a single grid image without overlapping.

    :param image_list: List of individual images to combine.
    :param grid_size: Tuple representing the number of rows and columns in the grid.
    :param image_size: Size of each individual image.
    :return: Combined Image.
    """
    rows, cols = grid_size
    img_width, img_height = image_size

    # Calculate the size of the final combined image
    combined_width = cols * img_width
    combined_height = rows * img_height
    combined_image = Image.new('RGB', (combined_width, combined_height), color='white')

    # Paste each image into the grid, respecting its size
    for i, img in enumerate(image_list):
        # Ensure each image is resized to the expected image size to avoid overlapping
        img = img.resize(image_size)

        row = i // cols
        col = i % cols
        x_offset = col * img_width
        y_offset = row * img_height

        # Paste the image into the correct position
        combined_image.paste(img, (x_offset, y_offset))

    return combined_image
