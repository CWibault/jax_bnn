from functools import partial

import jax
import jax.numpy as jnp
from flax.training import train_state
import numpy as np
import optax


class TrainState(train_state.TrainState):
  key: jax.Array


class VMapCallbackLogEvalMLP:
    def __init__(self):
        self.history = {"epoch": [], "mean_train_mse_loss": [], "mean_val_mse_loss": []}

    def __call__(self, log_dict):
        self.history['epoch'].append(log_dict['epoch'])
        self.history['mean_train_mse_loss'].append(log_dict['mean_train_mse_loss'])
        self.history['mean_val_mse_loss'].append(log_dict['mean_val_mse_loss'])

    def get_combined_history(self):
        return {'epoch': np.array(self.history['epoch']),
                'mean_train_mse_loss': np.array(self.history['mean_train_mse_loss']),
                'mean_val_mse_loss': np.array(self.history['mean_val_mse_loss'])}


def train_mlp(config, train_data, test_data, model, rng, calculate_weights=None):
    """
    --- train MLP in JAX --- 
    args:
        config: configuration dictionary
        train_data: tuple of training inputs and targets (X_train, Y_train)
        test_data: tuple of testing inputs and targets (X_test, Y_test)
        model: JAX model to train
        rng: random number generator
        calculate_weights: optional function to reweight loss function
    returns:
        final_params: trained model parameters
    """

    @partial(jax.jit, static_argnums=(3, 4, 5, 6))
    def train_step_MLP(state, x, y, dropout_key, update_gradients=True, l1_regularisation_weight=0.0, l2_regularisation_weight=0.0):
        """Train for a single step."""
        dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
        def loss_fn(params):
            pred = state.apply_fn(params, x, training=True, rngs={'dropout': dropout_train_key})
            if calculate_weights is not None:
                weights, _ = calculate_weights(y)
                mse_loss = jnp.mean(weights * jnp.mean(optax.l2_loss(pred, y), axis=1))
            else:
                mse_loss = jnp.mean(optax.l2_loss(pred, y))
            l1_reg = sum(jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params))
            l2_reg = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
            loss = mse_loss + l1_regularisation_weight * l1_reg + l2_regularisation_weight * l2_reg
            return loss
        
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        if update_gradients == True: state = state.apply_gradients(grads=grads)
        return state, loss


    @partial(jax.jit, static_argnums=(3, 4))
    def test_step_MLP(state, x, y, l1_regularisation_weight=0.0, l2_regularisation_weight=0.0):
        """Evaluate model on a single step."""
        def loss_fn(params):
            pred = state.apply_fn(params, x, training=False)
            if calculate_weights is not None:
                weights, _ = calculate_weights(y)
                mse_loss = jnp.mean(weights * jnp.mean(optax.l2_loss(pred, y), axis=1))
            else:
                mse_loss = jnp.mean(optax.l2_loss(pred, y))
            l1_reg = sum(jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params))
            l2_reg = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
            loss = mse_loss + l1_regularisation_weight * l1_reg + l2_regularisation_weight * l2_reg
            return loss
        loss = loss_fn(state.params)
        return loss

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
    rng, params_key, dropout_key = jax.random.split(key = rng, num=3)
    dummy_x = jnp.zeros((config["BATCH_SIZE"], X_train.shape[-1]))
    params = model.init(params_key, dummy_x, training=False)

    # --- initialise optimiser state --- 
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=config["LR"]))
    state = TrainState.create(apply_fn=model.apply, params=params, key=dropout_key, tx=tx)

    # --- train for certain number of gradient update steps --- 
    if "NUMBER_OF_EPOCHS" in config: 
        number_of_epochs = config["NUMBER_OF_EPOCHS"]
    else: 
        updates_per_epoch = dataset_length // config["BATCH_SIZE"]
        total_update_steps = config["NUMBER_OF_UPDATE_STEPS"]
        number_of_epochs = total_update_steps // updates_per_epoch
        jax.debug.print("NUMBER OF EPOCHS: {}", number_of_epochs)

    # --- calculate number of training and validation batches --- 
    num_train_batches = len(X_train) // config["BATCH_SIZE"]
    num_val_batches = len(X_test) // config["BATCH_SIZE"]
    truncated_train_size = num_train_batches * config["BATCH_SIZE"]
    truncated_val_size = num_val_batches * config["BATCH_SIZE"]


    def train_eval_single_epoch(state, rng, epoch_idx):  

        # --- prepare batches by shuffling and truncating --- 
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
        
        state, _ = jax.lax.scan(lambda carry, batch: train_step_MLP(carry, 
                                                                    batch[0], batch[1], dropout_key, True,
                                                                    config["L1_REGULARISATION_WEIGHT"], 
                                                                    config["L2_REGULARISATION_WEIGHT"]), 
                                                                    state, (x_train_batches, y_train_batches))
        train_mse_loss = jax.vmap(test_step_MLP, in_axes=(None, 0, 0, None, None))(state, 
                                                                                   x_train_batches, 
                                                                                   y_train_batches,
                                                                                   config["L1_REGULARISATION_WEIGHT"], 
                                                                                   config["L2_REGULARISATION_WEIGHT"])
        val_mse_loss = jax.vmap(test_step_MLP, in_axes=(None, 0, 0, None, None))(state, x_val_batches, y_val_batches, config["L1_REGULARISATION_WEIGHT"], config["L2_REGULARISATION_WEIGHT"])
        jax.debug.print("========================EPOCH {}========================", epoch_idx)
        jax.debug.print("TRAIN_MSE_LOSS:{}", jnp.mean(train_mse_loss))
        jax.debug.print("VAL_MSE_LOSS:{}", jnp.mean(val_mse_loss))
        return (state, rng), (jnp.mean(train_mse_loss), jnp.mean(val_mse_loss), epoch_idx)
        
    (state, rng), (mean_train_mse_loss, mean_val_mse_loss, epoch_idx) = jax.lax.scan(lambda carry, input: train_eval_single_epoch(carry[0], carry[1], input), (state, rng), jnp.arange(number_of_epochs))
    return mean_train_mse_loss, mean_val_mse_loss, epoch_idx, state.params