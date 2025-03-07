from flax.linen.initializers import zeros
from typing import Any, Tuple

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

def mean_init_zeros(key: PRNGKey, 
                    shape: Shape, 
                    dtype: Dtype) -> Array:
    """
    --- function to initialise the mean parameter to zero --- 
    """
    return zeros(key, shape, dtype)