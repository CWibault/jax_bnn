from flax.linen.initializers import ones 
import jax.numpy as jnp
from typing import Any, Tuple

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

def log_sigma_mean_init(key: Any, 
                        shape: Tuple[int, ...], 
                        dtype: Any) -> Array:
    """
    --- initialise the mean parameter to a small value --- 
    """
    return ones(key, shape, dtype) * jnp.log(0.1)