import jax.numpy as jnp
from typing import Tuple

def prior_std_xavier(shape: Tuple[int, ...]) -> float:
    """
    --- compute a prior standard deviation using Xavier initialization for a given shape --- 
    args:
        shape: tuple or list representing the shape of a parameter
    returns:
        standard deviation of the parameter
    """
    # --- remove dimensions of size 1 --- 
    filtered_shape = [dim for dim in shape if dim > 1] 
    if len(filtered_shape) > 1:
        fan_in, fan_out = filtered_shape[0], filtered_shape[1]
        stdv = jnp.sqrt(2 / (fan_in + fan_out)) 
    else:
        stdv = 1e-2 
    return stdv