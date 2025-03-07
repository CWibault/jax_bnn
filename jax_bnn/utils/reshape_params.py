import jax.numpy as jnp


def reshape_params(target: dict, 
                   source: dict) -> dict:
    """
    --- function to replace parameters of one dicionary with another --- 
    if the target is a dummy dictionary, then the source is essentially reshaped to match the target (but function works by replacing target values with source values and returning the target, rather than returning the reshaped source).
    any parameters in the source that are not in the target are ignored. 
    args:
        target: dictionary of parameters whose shape is to be matched
        source: dictionary of parameters to be reshaped
    returns:
        target: dictionary of parameters updated with values from the source dictionary
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            reshape_params(target[key], value)
        elif key in target:
            target[key] = jnp.reshape(value, jnp.shape(target[key]))
    return target