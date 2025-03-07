import jax.numpy as jnp
from typing import Tuple

def prior_std_reasonable(shape: Tuple[int, ...]) -> float:
    """
    --- compute a reasonable prior standard deviation for each parameter in an array --- 
    args:
        shape: tuple or list representing the shape of a parameter
    returns:
        standard deviation of the parameter
    """
    # --- remove dimensions of size 1 --- 
    filtered_shape = [dim for dim in shape if dim > 1]  
    if len(filtered_shape) > 1:
        stdv = 1 / jnp.sqrt(jnp.prod(jnp.array(filtered_shape[1:])))
    else:
        stdv = 1e-2
    # --- clip standard deviation to 0.1 --- 
    stdv = jnp.clip(stdv, a_min=None, a_max=0.1) 
    return stdv