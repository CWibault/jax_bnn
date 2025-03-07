import jax.numpy as jnp
import json 
import numpy as np

def save_history(history: dict, file_path: str):
    """
    --- function to save history to a json file. Differentiates between single dictionary and list of dictionaries --- 
    args: 
        history: dict - history dictionary with values converted to jax arrays
        file_path: str - path to the json file
    """
    if isinstance(history, dict): 
        serializable_dict = {k: (v.tolist() if isinstance(v, (np.ndarray, jnp.ndarray)) else v)
                            for k, v in history.items()}
        with open(file_path, "w") as f:
            json.dump(serializable_dict, f)
    else: 
        serializable_history = []
        for log_dict in history:
            serializable_log_dict = {k: (v.tolist() if isinstance(v, (np.ndarray, jnp.ndarray)) else v)
                                    for k, v in log_dict.items()}
            serializable_history.append(serializable_log_dict)
        with open(file_path, "w") as f:
            json.dump(serializable_history, f)
    return 