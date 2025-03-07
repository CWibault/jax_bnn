import jax.numpy as jnp
import json

def load_history(file_path: str) -> dict:
    """
    --- function to load history from a json file. Handles both list of dictionaries and single dictionary formats --- 
    args: file_path: str - path to the json file
    returns: history dictionary with values converted to jax arrays
    """
    with open(file_path, "r") as f:
        serializable_history = json.load(f)

    if isinstance(serializable_history, dict):
        history = {k: jnp.array(v) if isinstance(v, list) else v
                   for k, v in serializable_history.items()}
    else:
        history = []
        for log_dict in serializable_history:
            recreated_log_dict = {k: (jnp.array(v) if isinstance(v, list) else v)
                                  for k, v in log_dict.items()}
            history.append(recreated_log_dict)

    return history