import jax.numpy as jnp
from typing import Optional


def negative_log_likelihood(Y: jnp.ndarray, 
        mu: jnp.ndarray, 
        log_sigma: Optional[jnp.ndarray] = None, 
        reduction: str = "mean") -> jnp.ndarray:
    """
    --- calculates the negative log-likelihood of the model given the data --- 
    args:
        Y: target values matrix (n x o)
        mu: predicted mean values matrix (n x o)
        log_sigma: log standard deviation matrix (n x o)
        reduction: reduction method for the negative log-likelihood. Can be "mean" or "sum".
    returns:
        mean or sum of the negative log-likelihood
    """
    if log_sigma is None:
        vars = 0.01 * jnp.ones(Y.shape)
    else:
        vars = (jnp.exp(log_sigma))**2
    assert vars.shape == Y.shape, "Shape mismatch between variance and Y"

    log_likes = -0.5 * jnp.sum(((Y - mu) ** 2) / vars + jnp.log(2 * jnp.pi * vars), axis=-1)

    if reduction == "mean":
        return -jnp.mean(log_likes)
    elif reduction == "sum":
        return -jnp.sum(log_likes)
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")