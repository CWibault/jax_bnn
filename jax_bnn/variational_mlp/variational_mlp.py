from jax_bnn.variational_mlp.variational_modules.variational_parameter import VariationalParameter
from jax_bnn.variational_mlp.variational_modules.variational_dense import VariationalDense
from jax_bnn.variational_mlp.variational_utils.log_sigma_mean_init import log_sigma_mean_init
from jax_bnn.variational_mlp.variational_utils.log_sigma_rho_init import log_sigma_rho_init
from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence, Optional, Tuple


class VariationalMLP(nn.Module):
    """
    --- variational multi-layer perceptron --- 
    args:
        input_dim: the dimension of the input
        output_dim: the dimension of the output
        hidden_layers: the dimensions of the hidden layers
        learn_sigma: whether to learn the standard deviation of the output layer
    """

    input_dim: int
    output_dim: int
    hidden_layers: Sequence[int] = (256, 256, 256) 
    learn_sigma: bool = True

    def setup(self):
        self.layers = [VariationalDense(features=hidden_dim) for hidden_dim in self.hidden_layers]
        self.output_layer = VariationalDense(features=self.output_dim)
        if self.learn_sigma:
            self.log_sigma = VariationalParameter(features=self.output_dim, zero_mean=False, mean_init=log_sigma_mean_init, rho_init=log_sigma_rho_init)

    def __call__(self, X: jax.Array, deterministic: Optional[bool] = True, rng: Optional[jax.random.PRNGKey] = None) -> Tuple[jax.Array, jax.Array]:
        """
        --- forward pass through the variational MLP --- 
        args:
            X: input array of shape (batch_size, input_dim)
            deterministic: whether to sample or use deterministic outputs
            rng: random number generator key for sampling
        returns:
            tuple of predicted mean (shape: (batch_size, output_dim)) and 
            log variance (shape: (batch_size, output_dim))
        """
        if deterministic: 
            for i, layer in enumerate(self.layers):
                X = layer(X, deterministic, rng)
                X = nn.relu(X)
            X = self.output_layer(X, deterministic, rng)
            if self.learn_sigma:
                log_sigma = self.log_sigma(shape=(self.output_dim,), deterministic=deterministic, rng=rng)
            else:
                log_sigma = jnp.log(0.01) * jnp.ones((X.shape)) 
        else:
            assert rng is not None, "rng is required for non-deterministic sampling"
            rngs = jax.random.split(rng, len(self.layers) + 2)
            for i, layer in enumerate(self.layers):
                X = layer(X, deterministic, rngs[i])
                X = nn.relu(X)
            X = self.output_layer(X, deterministic, rngs[-2])
            if self.learn_sigma:
                log_sigma = self.log_sigma(shape=(self.output_dim,), deterministic=deterministic, rng=rngs[-1])
            else:
                log_sigma = jnp.log(0.01) * jnp.ones((X.shape)) 
        return X, log_sigma