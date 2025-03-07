import argparse
from flax.core.frozen_dict import unfreeze
import jax
import jax.numpy as jnp
from jax_bnn.log_utils.save_history import save_history
from jax_bnn.utils.normalise import create_gaussian_normalisation_functions
from jax_bnn.utils.reshape_params import reshape_params
from jax_bnn.utils.sample_tree_diag import sample_tree_diag
from jax_bnn.variational_mlp.variational_mlp import VariationalMLP
from jax_bnn.variational_mlp.mlp import MLP
from jax_bnn.variational_mlp.train_variational_mlp import train_variational_mlp
from jax_bnn.variational_mlp.train_mlp import train_mlp
from jax_bnn.variational_mlp.variational_utils.log_posterior_distribution import log_posterior_distribution
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import shutil


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='test_vi_2d')
    parser.add_argument("--n_train_data", type=int, default=5000)
    parser.add_argument("--n_test_data", type=int, default=1000)
    parser.add_argument("--n_posterior_samples", type=int, default=10)
    return parser.parse_args()


def test_vi_2d(args):    
        
    # --- load config --- 
    with open(f"configs/{args.task}.yml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["ENV_NAME"] = args.task
    base_path = f"runs/{args.task}/seed_{config['SEED']}"
    os.makedirs(base_path, exist_ok=True)
    shutil.copy(f"configs/{args.task}.yml", f"{base_path}/config.yml")
        
    np.random.seed(config["SEED"])
    rng = jax.random.PRNGKey(config["SEED"])

    # --- function to be approximated by neural network --- 
    def generate_data(n_samples):
        """Generate 2D datapoints with sum of Gaussian blobs."""
        centers = np.array([[1.0, 1.0], [-1.0, -1.0], [2.0, -2.0]])
        scales = np.array([0.5, 0.7, 0.3])
        x = np.random.uniform(-3, 3, size=(n_samples, 2))  
        y = np.zeros((n_samples, 1))
        for center, scale in zip(centers, scales):
            y += np.exp(-np.sum(((x - center) / scale) ** 2, axis=1, keepdims=True))
        return x, y
        
    # --- initialise and normalise dataset --- 
    train_x, train_y = generate_data(args.n_train_data)
    test_x, test_y = generate_data(args.n_test_data)
    test_x = test_x
    test_y  = test_y

    normalise_inputs, _, _, _ = create_gaussian_normalisation_functions(samples=train_x)
    normalise_outputs, denormalise_outputs, _, _ = create_gaussian_normalisation_functions(samples=train_y)
    normalised_train_x = jnp.array(normalise_inputs(train_x))
    normalised_train_y = jnp.array(normalise_outputs(train_y))
    normalised_test_x = jnp.array(normalise_inputs(test_x))
    normalised_test_y = jnp.array(normalise_outputs(test_y))
            
    # --- initialise variational mlp --- 
    variational_mlp = VariationalMLP(input_dim = 2,
                                            output_dim = 1,
                                            hidden_layers = config["MODEL"]["HIDDEN_LAYERS"],
                                            learn_sigma = config["VARIATIONAL_INFERENCE"]["LEARN_SIGMA"])

    # --- optionally warm start variational inference --- 
    if config["VARIATIONAL_INFERENCE"]["WARM_START"] == False: 
        warm_start_params = None
    else:
        mlp = MLP(input_dim = 2,
                        output_dim = 1,
                        hidden_layers = config["MODEL"]["HIDDEN_LAYERS"])
        rng, _rng = jax.random.split(rng)
        _, warm_start_params = train_mlp(config["VARIATIONAL_INFERENCE"]["WARM_START"],
                                                        (normalised_train_x, normalised_train_y),
                                                        (normalised_test_x, normalised_test_y), 
                                                        mlp, 
                                                        _rng)

    # --- train variational mlp --- 
    rng, _rng = jax.random.split(rng)
    eval_history, variational_mlp_params, _, _ = train_variational_mlp(config["VARIATIONAL_INFERENCE"],
                                                                           (normalised_train_x, normalised_train_y),
                                                                           (normalised_test_x, normalised_test_y),
                                                                           variational_mlp,
                                                                           _rng, 
                                                                           warm_start_params)
    if config["LOG"] == True:
        save_history(eval_history, f"{base_path}/eval_history.json")
        mean_tree, cov_tree = log_posterior_distribution(variational_mlp_params, base_path, eval_history)
    else:
        mean_tree, cov_tree = log_posterior_distribution(variational_mlp_params)

    # --- visualise posterior predictions --- 
    model = MLP(input_dim = 2,
                    output_dim = 1,
                    hidden_layers = config["MODEL"]["HIDDEN_LAYERS"])
    dummy_params = model.init(rng, jnp.ones((1, 2)))

    # --- create grid for visualisation --- 
    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()] 

    for _ in range(args.n_posterior_samples):
        rng, _rng = jax.random.split(rng)
        sampled_params = sample_tree_diag(mean_tree, cov_tree, _rng)
        sampled_params = reshape_params(unfreeze(dummy_params), sampled_params)
            
    # --- make predictions --- 
    normalised_inputs = normalise_inputs(jnp.array(grid_points))
    normalised_predictions = model.apply(sampled_params, normalised_inputs).reshape(100, 100)
    predictions = denormalise_outputs(normalised_predictions)

    # --- plot 1: posterior predictions --- 
    plt.contourf(grid_x, grid_y, predictions, levels=20, cmap="viridis", alpha=0.7)
    plt.scatter(test_x[:, 0], test_x[:, 1], c=test_y.flatten(), cmap="coolwarm", edgecolors="black", label="Test Points")
    plt.colorbar(label="Predicted Value")
    plt.title("Posterior Predictions (2D)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{base_path}/posterior_predictions.png")
    plt.close()
        
    # --- plot 2: training loss v epochs --- 
    plt.figure(figsize=(10, 5))
    plt.plot(eval_history["epoch"], eval_history["mean_train_loss"], '-', linewidth=2, color="blue", label="Loss")
    plt.plot(eval_history["epoch"], eval_history["mean_train_kl_loss"], '-', linewidth=2, color="red", label="Prior Loss")
    plt.plot(eval_history["epoch"], eval_history["mean_train_nll_loss"], '-', linewidth=2, color="green", label="NLL Loss")
    plt.title('Training Loss v Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss per Train/Validation datapoint')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{base_path}/loss_v_epochs.png")
    plt.close()

    # --- plot 3: posterior variance v epochs --- 
    plt.figure(figsize=(10, 5))
    plt.plot(eval_history["epoch"], eval_history["post_var"], '-', linewidth=2, color="blue", label="Posterior Variance")
    plt.title('Posterior Variance v Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Posterior Variance')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{base_path}/post_var_v_epochs.png")
    plt.close()
    return


if __name__ == "__main__":
    args = get_args()
    test_vi_2d(args)

    