from flax.linen.initializers import ones
import jax.numpy as jnp
from .prior_std_reasonable import prior_std_reasonable as prior_std


def rho_init(key, shape, dtype) -> jnp.ndarray:
    """
    --- initialise the rho for each parameter of a jnp array based on the prior standard deviation --- 
    args:
        key: JAX random key
        shape: shape of parameter
        dtype: data type of parameter
    returns: initialised rho for each parameter
    """
    stdv = prior_std(shape)
    init_rho = jnp.log(jnp.exp(stdv) - 1)
    return ones(key, shape, dtype) * init_rho