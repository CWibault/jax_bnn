from flax.core.frozen_dict import unfreeze, freeze
from flax.training import train_state
from functools import partial
import jax
import jax.numpy as jnp
from jax_bnn.variational_mlp.variational_utils.set_prior import set_prior
from jax_bnn.variational_mlp.variational_utils.mean_init_warm_start import mean_init_warm_start
from jax_bnn.variational_mlp.variational_utils.negative_log_likelihood import negative_log_likelihood
from jax_bnn.variational_mlp.variational_utils.track_posterior_variance import track_posterior_variance
from jax_bnn.variational_mlp.variational_utils.variationalise_params import variationalise_params
import numpy as np
import optax
from typing import Optional


class VMapCallbackLogEvalVariationalMLP:
    """
    Callback class for logging evaluation metrics during training of a Variational MLP.
    """
    def __init__(self):
        self.history = {"epoch": [], "mean_train_loss": [], "mean_train_nll_loss": [], "mean_train_kl_loss": [], "mean_val_loss": [], "mean_val_mse_loss": [], "post_var": []}

    def __call__(self, log_dict):
        self.history['epoch'].append(log_dict['epoch'])
        self.history['mean_train_loss'].append(log_dict['mean_train_loss'])
        self.history['mean_train_nll_loss'].append(log_dict['mean_train_nll_loss'])
        self.history['mean_train_kl_loss'].append(log_dict['mean_train_kl_loss'])
        self.history['mean_val_loss'].append(log_dict['mean_val_loss'])
        self.history['mean_val_mse_loss'].append(log_dict['mean_val_mse_loss'])
        self.history['post_var'].append(log_dict['post_var'])

    def get_combined_history(self):
        return {'epoch': np.array(self.history['epoch']),
            'mean_train_loss': np.array(self.history['mean_train_loss']),
            'mean_train_nll_loss': np.array(self.history['mean_train_nll_loss']),
            'mean_train_kl_loss': np.array(self.history['mean_train_kl_loss']),
            'mean_val_loss': np.array(self.history['mean_val_loss']),
            'mean_val_mse_loss': np.array(self.history['mean_val_mse_loss']),
            'post_var': np.array(self.history['post_var'])}


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def train_step_VariationalMLP(state, x, y, rng, dataset_length, kl_weight, prior_loss_function, deterministic=False, update_gradients=True):
    """Train for a single step."""
    rng, _rng = jax.random.split(rng)
    def loss_fn(params):
        rng_keys = jax.random.split(_rng, len(x))
        def apply_fn_independent(x, rng):
            pred, log_sigma = state.apply_fn(params, x, deterministic, rng)
            return pred, log_sigma
        pred, log_sigma = jax.vmap(apply_fn_independent)(x, rng_keys)
        loss_nll = negative_log_likelihood(y, pred, log_sigma)
        loss_kl = kl_weight * prior_loss_function(params) / dataset_length
        return loss_nll + loss_kl
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    if update_gradients: state = state.apply_gradients(grads=grads) 
    return (state, rng), None


@partial(jax.jit, static_argnums=(4, 5, 6, 7))
def test_step_VariationalMLP(state, x, y, rng, dataset_length, kl_weight, prior_loss_function, deterministic=False):
    """Test for a single step."""
    def loss_fn(params):
        rng_keys = jax.random.split(rng, len(x))
        def apply_fn_independent(x, rng):
            pred, log_sigma = state.apply_fn(params, x, deterministic, rng)
            return pred, log_sigma
        pred, log_sigma = jax.vmap(apply_fn_independent)(x, rng_keys)
        loss_nll = negative_log_likelihood(y, pred, log_sigma)
        loss_kl = kl_weight * prior_loss_function(params) / dataset_length
        return loss_nll + loss_kl, loss_nll, loss_kl
    loss, loss_nll, loss_kl = loss_fn(state.params)
    return loss, loss_nll, loss_kl


@jax.jit
def test_step_VariationalMLP_MSE(state, x, y):
    """Test for a single step."""
    def loss_fn(params):
        pred, log_sigma = state.apply_fn(params, x, deterministic=True, rng=None)
        loss = jnp.mean(optax.l2_loss(pred, y))
        return loss, log_sigma
    loss, log_sigma = loss_fn(state.params)
    return loss, log_sigma


def train_variational_mlp(config, train_data, test_data, model, rng, warm_start_params: Optional[dict] = None):
    """
    --- train variational MLP in JAX, tracking average epoch loss, evaluating test loss, and stopping based on convergence or overfitting --- 
    args:
        config: configuration dictionary
        train_data: tuple of training inputs and targets (X_train, Y_train)
        test_data: tuple of testing inputs and targets (X_test, Y_test)
        model: JAX model to train
        rng: random number generator
        warm_start_params: optional dictionary of parameters to warm start the model
    returns:
        final_params: trained model parameters
    """
    # --- for logging --- 
    vmap_callback_log_eval = VMapCallbackLogEvalVariationalMLP()

    # --- unpack training and testing data --- 
    X_train, Y_train = train_data[0], train_data[1]
    X_test, Y_test = test_data[0], test_data[1]
    dataset_length = len(X_train)
    jax.debug.print("==================================")
    jax.debug.print("X_TRAIN SHAPE:{}", X_train.shape)
    jax.debug.print("Y_TRAIN SHAPE:{}", Y_train.shape)
    jax.debug.print("DATASET_LENGTH:{}", dataset_length)
    jax.debug.print("==================================")

    # --- initialise model parameters --- 
    rng, _rng = jax.random.split(rng)
    dummy_x = jnp.zeros((config["BATCH_SIZE"], X_train.shape[-1]))
    params = model.init(_rng, dummy_x, deterministic=False, rng=_rng)
    if warm_start_params is not None:
        warm_start_params = variationalise_params(unfreeze(warm_start_params))
        params_unfrozen = unfreeze(params)
        mean_init_warm_start(params_unfrozen, warm_start_params)
        params = freeze(params_unfrozen)

    # --- set prior --- 
    prior_loss_function = set_prior(config["PRIOR_TYPE"])

    # --- initialise optimiser state --- 
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=config["LR"]))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # --- calculate number of training and validation batches --- 
    updates_per_epoch = dataset_length // config["BATCH_SIZE"]
    total_update_steps = config["MAX_UPDATE_STEPS"]
    number_of_epochs = total_update_steps // updates_per_epoch

    num_train_batches = len(X_train) // config["BATCH_SIZE"]
    num_val_batches = len(X_test) // config["BATCH_SIZE"]
    truncated_train_size = num_train_batches * config["BATCH_SIZE"]
    truncated_val_size = num_val_batches * config["BATCH_SIZE"]

    def train_eval_single_epoch(state, rng, epoch_idx):  
        # --- prepare dataset --- 
        rng, _rng = jax.random.split(rng)
        shuffle_indices = jax.random.permutation(_rng, len(X_train))
        rng, _rng = jax.random.split(rng)
        val_shuffle_indices = jax.random.permutation(_rng, len(X_test))
        shuffled_x_train = X_train[shuffle_indices, ...]
        shuffled_y_train = Y_train[shuffle_indices, ...]
        shuffled_x_test = X_test[val_shuffle_indices, ...]
        shuffled_y_test = Y_test[val_shuffle_indices, ...]
        x_train_batches = shuffled_x_train[:truncated_train_size].reshape(-1, config["BATCH_SIZE"], *X_train.shape[1:])
        y_train_batches = shuffled_y_train[:truncated_train_size].reshape(-1, config["BATCH_SIZE"], *Y_train.shape[1:])
        x_val_batches = shuffled_x_test[:truncated_val_size].reshape(-1, config["BATCH_SIZE"], *X_test.shape[1:])
        y_val_batches = shuffled_y_test[:truncated_val_size].reshape(-1, config["BATCH_SIZE"], *Y_test.shape[1:])
        
        rng, _rng = jax.random.split(rng)
        (state, rng), _ = jax.lax.scan(lambda carry, batch: train_step_VariationalMLP(carry[0], batch[0], batch[1], carry[1], dataset_length, config["KL_WEIGHT"], prior_loss_function), (state, _rng), (x_train_batches, y_train_batches))
        rng, _rng = jax.random.split(rng)
        train_rngs = jax.random.split(_rng, num_train_batches)
        train_loss, train_nll_loss, train_kl_loss = jax.vmap(test_step_VariationalMLP, in_axes=(None, 0, 0, 0, None, None, None))(state, x_train_batches, y_train_batches, train_rngs, dataset_length, config["KL_WEIGHT"], prior_loss_function)
        rng, _rng = jax.random.split(rng)
        val_rngs = jax.random.split(_rng, num_val_batches)
        val_loss, _, _ = jax.vmap(test_step_VariationalMLP, in_axes=(None, 0, 0, 0, None, None, None))(state, x_val_batches, y_val_batches, val_rngs, dataset_length, config["KL_WEIGHT"], prior_loss_function)
        val_mse_loss, _ = jax.vmap(test_step_VariationalMLP_MSE, in_axes=(None, 0, 0))(state, x_val_batches, y_val_batches)

        post_var = track_posterior_variance(state.params)
        log_dict = {"epoch": epoch_idx, 
                    "mean_train_loss": jnp.mean(train_loss), 
                    "mean_train_nll_loss": jnp.mean(train_nll_loss),
                    "mean_train_kl_loss": jnp.mean(train_kl_loss),
                    "mean_val_loss": jnp.mean(val_loss), 
                    "mean_val_mse_loss": jnp.mean(val_mse_loss), 
                    "post_var": post_var}
        jax.debug.callback(vmap_callback_log_eval, log_dict)
        jax.debug.print("========================EPOCH {}========================", epoch_idx)
        jax.debug.print("mean_train_loss:{}", jnp.mean(train_loss))
        jax.debug.print("mean_train_nll_loss:{}", jnp.mean(train_nll_loss))
        jax.debug.print("mean_train_kl_loss:{}", jnp.mean(train_kl_loss))
        jax.debug.print("mean_val_loss:{}", jnp.mean(val_loss))
        jax.debug.print("mean_val_mse_loss:{}", jnp.mean(val_mse_loss))
        jax.debug.print("post_var:{}", post_var)
        return (state, rng), (jnp.mean(train_loss), jnp.mean(train_nll_loss), jnp.mean(train_kl_loss), jnp.mean(val_loss), jnp.mean(val_mse_loss), post_var)
        
    (state, rng), metrics = jax.lax.scan(lambda carry, input: train_eval_single_epoch(carry[0], carry[1], input), (state, rng), jnp.arange(number_of_epochs))
    _, _, _, _, mean_val_mse_loss, post_var = metrics
    eval_history = vmap_callback_log_eval.get_combined_history()
    return eval_history, state.params, mean_val_mse_loss, post_var