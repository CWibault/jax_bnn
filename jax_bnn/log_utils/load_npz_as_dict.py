import numpy as np


def load_npz_as_dict(file_path: str) -> dict:
    """
    --- function to convert npz file to mutable dictionary --- 
    args: file_path: str - path to npz file
    returns: dictionary of npz file
    """
    npz = np.load(file_path)

    return {key: npz[key] for key in npz.files}