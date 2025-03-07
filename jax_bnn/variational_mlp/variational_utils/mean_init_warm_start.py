import jax.numpy as jnp

def mean_init_warm_start(target: dict, 
                        source: dict) -> dict:
    """
    --- function required for implementing warm starting of the variational inference --- 
    recursively updates the target dictionary with values from the source dictionary.
    retains keys in the target dictionary that are not present in the source dictionary.
    args:
        target: dictionary of parameters to be updated
        source: dictionary of parameters to be used for warm starting
    returns:
        target: dictionary of parameters updated with values from the source dictionary
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            mean_init_warm_start(target[key], value)
        elif key in target:
            # reshape value to match target key
            target[key] = jnp.reshape(value, jnp.shape(target[key]))
        else:
            target[key] = value
    return target