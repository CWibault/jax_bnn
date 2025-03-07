from jax_bnn.log_utils.save_pytree import save_pytree
from flax.core.frozen_dict import unfreeze
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from typing import Optional


def log_posterior_distribution(variational_params: dict, 
                               base_path: Optional[str] = None, 
                               save_posterior: bool = False) -> tuple[dict, dict]:
    """
    --- compute the posterior distribution based on the variational parameters --- 
    args: 
        variational_params: nested dictionary containing variational parameters, where each leaf node has 'mean' and 'rho'
        base_path: str - path to save the posterior distribution
        save_posterior: bool - whether to save the posterior distribution
    returns:
        means_pytree: nested dictionary containing the means of the variational parameters
        cov_pytree: nested dictionary containing the covariances of the variational parameters
    """
    def extract_mean(param):
        if isinstance(param, dict) and "mean" in param and "rho" in param:
            return param["mean"]
    
    def extract_cov(param):
        if isinstance(param, dict) and "mean" in param and "rho" in param:
            cov = jnp.log(1 + jnp.exp(param["rho"])) ** 2
            return cov
        
    def is_leaf(node):
        return isinstance(node, dict) and "mean" in node and "rho" in node

    # Extract means and rhos as trees
    means_pytree = tree_map(extract_mean, variational_params, is_leaf=is_leaf)
    cov_pytree = tree_map(extract_cov, variational_params, is_leaf=is_leaf)

    # Flatten and concatenate means and rhos to single array
    means_flat = [jnp.ravel(arr) for arr in jax.tree_util.tree_flatten(means_pytree)[0] if arr is not None]
    cov_flat = [jnp.ravel(arr) for arr in jax.tree_util.tree_flatten(cov_pytree)[0] if arr is not None]
    posterior_mean = jnp.concatenate(means_flat)
    posterior_cov = jnp.concatenate(cov_flat)

    jax.debug.print('POSTERIOR MEAN SHAPE:{} ----- POSTERIOR COV SHAPE:{}', posterior_mean.shape, posterior_cov.shape)

    if save_posterior == True:
        assert description is not None and base_path is not None, "Description and base_path are required to save posterior distribution"
        save_pytree(unfreeze(means_pytree), f"{base_path}/posterior_distribution_mean.json")
        save_pytree(unfreeze(cov_pytree), f"{base_path}/posterior_distribution_cov.json")

    return means_pytree, cov_pytree
