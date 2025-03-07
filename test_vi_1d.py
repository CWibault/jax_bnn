import argparse
from flax.core.frozen_dict import unfreeze
import jax
import jax.numpy as jnp
from jax_bnn.log_utils.save_history import save_history
from jax_bnn.utils.reshape_params import reshape_params
from jax_bnn.utils.sample_tree_diag import sample_tree_diag
from jax_bnn.variational_mlp.mlp import MLP
from jax_bnn.variational_mlp.variational_mlp import VariationalMLP
from jax_bnn.variational_mlp.train_variational_mlp import train_variational_mlp
from jax_bnn.variational_mlp.variational_utils.log_posterior_distribution import log_posterior_distribution
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import yaml


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='test_vi_1d')
    parser.add_argument("--n_train_data", type=int, default=5000)
    parser.add_argument("--n_test_data", type=int, default=1000)
    parser.add_argument("--n_posterior_samples", type=int, default=10)
    return parser.parse_args()


def test_vi(args):  

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
        """ Generate n_samples regression datapoints """
        x = np.random.normal(size=(n_samples, 1))
        y = np.cos(x * 3)
        return x, y
            
    # --- initialise and normalise dataset --- 
    train_x, train_y = generate_data(args.n_train_data)
    train_y = train_y / np.std(train_y)
    test_x, test_y = generate_data(args.n_test_data)
    test_y = test_y / np.std(test_y)

    training_inputs = jnp.array(train_x)
    training_targets = jnp.array(train_y)
    testing_inputs = jnp.array(test_x)
    testing_targets = jnp.array(test_y)
        
    # --- initialise variational mlp --- 
    variational_mlp = VariationalMLP(input_dim = 1,
                                           output_dim = 1,
                                           hidden_layers = config["MODEL"]["HIDDEN_LAYERS"],
                                           learn_sigma = config["VARIATIONAL_INFERENCE"]["LEARN_SIGMA"])
    
    # --- train variational mlp --- 
    rng, _rng = jax.random.split(rng)
    eval_history, variational_mlp_params, _, _ = train_variational_mlp(config["VARIATIONAL_INFERENCE"],
                                                                      (training_inputs, training_targets),
                                                                      (testing_inputs, testing_targets),
                                                                       variational_mlp,
                                                                       _rng)
    
    if config["LOG"] == True:
        save_history(eval_history, f"{base_path}/eval_history.json")
        mean_tree, cov_tree = log_posterior_distribution(variational_mlp_params, base_path, eval_history)
    else:
        mean_tree, cov_tree = log_posterior_distribution(variational_mlp_params)
    
    # --- visualise posterior predictions --- 
    model = MLP(input_dim = 1,
                output_dim = 1,
                hidden_layers = config["MODEL"]["HIDDEN_LAYERS"])
    dummy_params = model.init(rng, jnp.ones((1, 1)))

    for _ in range(args.n_posterior_samples):
        rng, _rng = jax.random.split(rng)
        sampled_params = (sample_tree_diag(mean_tree, cov_tree, _rng))
        sampled_params = reshape_params(unfreeze(dummy_params), sampled_params)
        x = np.random.normal(size=(1000,))
        inputs = jnp.expand_dims(jnp.array(x), axis=-1)
        mean = model.apply(sampled_params, inputs)
        outputs = mean
        plt.plot(x, outputs, 'x', linewidth=2, color="blue")
        
    # --- plot 1: posterior predictions --- 
    plt.plot(testing_inputs, testing_targets, 'x', linewidth=2, color="black", label="Test Points")
    plt.title('Test Points and Posterior Predictions')
    plt.xlabel('x')
    plt.ylabel('y')
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
    test_vi(args)

    

