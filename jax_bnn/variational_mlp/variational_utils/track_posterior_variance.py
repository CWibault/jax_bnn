import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

@jax.jit
def track_posterior_variance(variational_params):
    """
    --- function to calculate and track posterior variance of variational parameters [can be used to determine convergence] --- 
    args:
        variational_params: dictionary of variational parameters
    returns:
        sum of the posterior variance of the variational parameters
    """
    def extract_cov(param):
        if isinstance(param, dict) and "mean" in param and "rho" in param:
            cov = jnp.log(1 + jnp.exp(param["rho"])) ** 2
            return cov
        else:
            return None
    
    def is_leaf(node):
        return isinstance(node, dict) and "mean" in node and "rho" in node

    cov_pytree = tree_map(extract_cov, variational_params, is_leaf=is_leaf)
    cov_flat = [jnp.ravel(arr) for arr in jax.tree_util.tree_flatten(cov_pytree)[0] if arr is not None]
    phi_cov = jnp.concatenate(cov_flat)
    posterior_var = jnp.sqrt(jnp.sum(phi_cov))
    return posterior_var