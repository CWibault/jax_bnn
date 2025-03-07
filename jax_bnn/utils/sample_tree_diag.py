import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_leaves, tree_structure, tree_unflatten


def sample_tree_diag(mean_tree, covariance_tree, rng):
    """
    --- function required for sampling model parameters from a posterior distribution. Parameters are sampled independently from a multivariate normal distribution. --- 
    args:
        mean_tree: dictionary containing parameter means
        covariance_tree: dictionary containing parameter covariances
        rng: PRNG key for sampling
    returns: 
        samples: dictionary containing sampled parameters
    """
    # --- generate different keys for each leaf to ensure independent sampling --- 
    num_leaves = len(tree_leaves(mean_tree))
    rngs = jax.random.split(rng, num_leaves) 
    key_tree = tree_unflatten(tree_structure(mean_tree), rngs)

    def sample_fn(mean, cov, rng):
        std = jnp.sqrt(cov)
        eps = jax.random.normal(rng, mean.shape)
        return mean + std * eps

    samples = tree_map(sample_fn, mean_tree, covariance_tree, key_tree)
    return samples