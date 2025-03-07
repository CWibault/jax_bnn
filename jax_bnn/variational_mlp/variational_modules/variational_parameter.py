from jax_bnn.variational_mlp.variational_utils.mean_init_zeros import mean_init_zeros
from jax_bnn.variational_mlp.variational_utils.rho_init import rho_init
from flax import linen as nn
from flax.linen.initializers import lecun_normal
import jax
import jax.numpy as jnp
from typing import Any, Tuple, Callable, Optional

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any 
Array = Any


class VariationalParameter(nn.Module):
    """
    --- variational parameter class: splits each parameter into a mean and a standard deviation: param = mean + std * eps --- 
    If the parameter is called deterministically, the mean is returned, otherwise a sample, eps, is drawn from a normal distribution with mean 0 and standard deviation 1. 
    The standard deviation is parameterised as log(1 + exp(rho)) to ensure positivity.
    args: 
        features: int - number of features
        zero_mean: if true, the mean is initialised to zero
        mean_init: initialisation function for the mean parameter. NB: overwritten if zero_mean is true
        rho_init: initialisation function for the rho parameter 
    """
    features: int 
    zero_mean: bool = True 
    mean_init: Callable[[PRNGKey, Shape, Dtype], Array] = lecun_normal()
    rho_init: Callable[[PRNGKey, Shape, Dtype], Array] = rho_init

    @nn.compact
    def __call__(self, shape: Array, deterministic: Optional[bool] = True, rng: Optional[PRNGKey] = None) -> Array:
        
        assert shape[-1] == self.features, "Shape mismatch between inputs and features"

        if not isinstance(shape, tuple):
            raise ValueError(f"Expected 'shape' to be a tuple, got {type(shape)}: {shape}")
        
        if self.zero_mean:
            mean_init = mean_init_zeros 
        else: 
            mean_init = self.mean_init
        mean = self.param("mean", mean_init, shape, jnp.float32)
        rho = self.param("rho", self.rho_init, shape, jnp.float32)
        std = jnp.log(1 + jnp.exp(rho)) 

        if deterministic:
            eps = jnp.zeros(mean.shape)
        else:
            assert rng is not None, "rng is required for non-deterministic sampling"
            eps = jax.random.normal(rng, mean.shape)
        return mean + std * eps