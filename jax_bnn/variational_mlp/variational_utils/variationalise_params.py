
def variationalise_params(d: dict) -> dict:
    """
    --- function required for implementing warm starting of the variational inference --- 
    augments each parameter of the parameter dictionary with a 'mean' key.
    args:
        d: dictionary of parameters
    returns:
        d: dictionary of parameters with a 'mean' key for each parameter
    """
    for key, value in d.items():
        # --- if the value is a dictionary, recurse --- 
        if isinstance(value, dict):
            variationalise_params(value)
        # --- if the value is a leaf, replace with nested dictionary ---
        else:
            d[key] = {"mean": value}
    return d