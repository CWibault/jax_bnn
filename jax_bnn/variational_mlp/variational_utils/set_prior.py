from jax_bnn.variational_mlp.variational_utils.prior_std_reasonable import prior_std_reasonable as prior_std
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from typing import Callable


def sub_prior_loss(variational_params):
    """
    --- function to compute exact KL divergence between gaussian prior and gaussian posterior --- 
    args:
        variational_params: dictionary containing variational parameters
    returns:
        KL divergence
    """
    def is_leaf(node):
        return isinstance(node, dict) and "mean" in node and "rho" in node
    
    def kl_divergence(param):
        mean = param["mean"]
        rho = param["rho"]
        std = jnp.log(1 + jnp.exp(rho))
        std_prior = prior_std(mean.shape)
        kl_div = jnp.sum(-jnp.log(std / std_prior) + (std**2 + mean**2) / (2 * std_prior**2) - 0.5)
        return kl_div
    
    kl_tree = tree_map(kl_divergence, variational_params, is_leaf=is_leaf)
    flat_kl_values = jnp.array(jax.tree_util.tree_flatten(kl_tree)[0])

    return jnp.sum(flat_kl_values)


def set_prior(prior_type: str, **prior_parameters) -> Callable:
    """
    --- function to set prior for variational parameters --- 
    args:
        prior_type: type of prior
        prior_parameters: parameters for prior
    returns:
        prior function
    """
    if prior_type == "gaussian":
        return sub_prior_loss
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")