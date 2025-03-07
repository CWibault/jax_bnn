from jax_bnn.variational_mlp.variational_modules.variational_parameter import VariationalParameter
import jax
from jax import lax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Tuple, Optional

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any 
Array = Any


class VariationalDense(nn.Module):
    """
    --- dense flax layer with variational parameters for the weights and biases --- 
    args: 
        features: int - number of output features
        use_bias: bool - whether to use bias
    """
    features: int
    use_bias: bool = True

    def setup(self):
        self.kernel = VariationalParameter(features=self.features)
        self.bias = VariationalParameter(features=self.features)
    
    def __call__(self, inputs: Any, deterministic: Optional[bool] = True, rng: Optional[PRNGKey] = None) -> Array:
        if deterministic:
            kernel = self.kernel((jnp.shape(inputs)[-1], self.features), deterministic, rng)
            if self.use_bias:
                bias = self.bias((self.features,), deterministic, rng)
            else:
                bias = None
        else:
            assert rng is not None, "rng is required for non-deterministic sampling"
            rng, _rng = jax.random.split(rng)
            kernel = self.kernel((jnp.shape(inputs)[-1], self.features), deterministic, rng)
            if self.use_bias:
                bias = self.bias((self.features,), deterministic, _rng)
            else:
                bias = None
        y = lax.dot_general(inputs, kernel,(((inputs.ndim - 1,), (0,)), ((), ())))
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y  