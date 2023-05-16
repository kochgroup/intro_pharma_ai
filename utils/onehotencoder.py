from rdkit import Chem
import numpy as np

def make_graph_features(m, atom_types):
    features = np.zeros((m.GetNumAtoms(), len(atom_types)))
    for i, a in enumerate(m.GetAtoms()):
        features[i] = a.GetAtomicNum()
    features = np.where(features == np.tile(atom_types, (m.GetNumAtoms(), 1)), 1, 0)
    return features

def onehotencode(mols):
    """Function to one-hot encode the atoms.

    Args:
        mols(list): List of Molecules in mol format.

        atom_types(list): List of atom types (atomic numbers). The length
            of the list will determine the first dimension of the feature
            matrix.
    """
    atom_types = []
    for m in mols:
        for atom in m.GetAtoms():
            if atom.GetAtomicNum() not in atom_types:
                atom_types.append(atom.GetAtomicNum())
    atom_types.sort()
    # print('atom_types',atom_types)

    onehotencodedatoms = [make_graph_features(m, atom_types) for m in mols]
    return onehotencodedatoms
