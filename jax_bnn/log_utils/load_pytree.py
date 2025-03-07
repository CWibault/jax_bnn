import jax.numpy as jnp
import json

def load_pytree(file_path: str) -> dict:
    """
    --- function to load pytree from json file --- 
    args: file_path: str - path to the json file
    returns: loaded pytree with original shapes restored
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    def serializable_to_pytree(obj):
        if isinstance(obj, dict) and "data" in obj and "shape" in obj:
            restored_array = jnp.array(obj["data"]).reshape(obj["shape"])
            if len(restored_array.shape) == 2 and restored_array.shape[1] == 1:
                restored_array = restored_array.squeeze(-1)  
            return restored_array
        elif isinstance(obj, list): 
            return [serializable_to_pytree(v) for v in obj]
        elif isinstance(obj, dict):  
            return {k: serializable_to_pytree(v) for k, v in obj.items()}
        else:
            return obj

    return serializable_to_pytree(data)