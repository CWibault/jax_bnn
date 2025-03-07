from flax.linen.initializers import ones 
from jax import numpy as jnp
from typing import Any, Tuple

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

def log_sigma_rho_init(key: PRNGKey, 
                       shape: Shape, 
                       dtype: Dtype) -> Array:
    """
    --- initialise rho to a small value --- 
    """
    stdv = 0.001
    init_rho = jnp.log(jnp.exp(stdv) - 1)
    return ones(key, shape, dtype) * init_rho