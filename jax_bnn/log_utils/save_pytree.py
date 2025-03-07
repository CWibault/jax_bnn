import jax.numpy as jnp
import json
import numpy as np

def save_pytree(pytree: dict, file_path: str):
    """
    --- function to save pytree to a json file --- 
    args: 
        pytree: dict - pytree containing jax or numpy arrays
        file_path: str - path to save the json file
    """
    def pytree_to_serializable(obj):
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            # --- save both data and shape information --- 
            return {"data": obj.tolist(), "shape": obj.shape} 
        elif isinstance(obj, dict):
            return {k: pytree_to_serializable(v) for k, v in obj.items()}  
        elif isinstance(obj, (list, tuple)):
            return [pytree_to_serializable(v) for v in obj] 
        else:
            return obj 

    serializable_pytree = pytree_to_serializable(pytree)
    with open(file_path, "w") as f:
        json.dump(serializable_pytree, f)
    return 