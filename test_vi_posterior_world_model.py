import argparse
from datasets.process_dataset import process_dataset
import flax
import jax
import jax.numpy as jnp
from jax_bnn.log_utils.load_history import load_history
from jax_bnn.log_utils.load_npz_as_dict import load_npz_as_dict
from jax_bnn.log_utils.save_history import save_history
from jax_bnn.utils.normalise import create_gaussian_normalisation_functions
from jax_bnn.utils.sample_dataset import sample_dataset
from jax_bnn.variational_mlp.mlp import MLP
from jax_bnn.variational_mlp.variational_mlp import VariationalMLP
from jax_bnn.variational_mlp.train_variational_mlp import train_variational_mlp
from jax_bnn.variational_mlp.variational_utils.log_posterior_distribution import log_posterior_distribution
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import yaml


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='CartPole-v1')
    parser.add_argument("--n_train_data_list", type=list, default=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000])
    parser.add_argument("--n_test_data", type=int, default=10000)    
    return parser.parse_args()


def learn_posterior_over_state_transition_model_parameters(args):
        
    """ 
    Example usage of BNN to learn a posterior over state transition model parameters in Reinforcement Learning
    Learning the posterior over parameters for different numbers of datapoints to demonstrate reducing effect of prior loss as N increases
    """

    # --- load config --- 
    with open(f"configs/{args.task}.yml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    base_path = f"runs/{config['ENV_NAME']}/{config['SEED']}"
    os.makedirs(base_path, exist_ok=True)
    shutil.copy(f"configs/{args.task}.yml", f"{base_path}/config.yml")
        
    np.random.seed(config["SEED"])
    rng = jax.random.PRNGKey(config["SEED"])
        
    # --- initialise and process dataset and required dimensions ---     
    unprocessed_dataset = load_npz_as_dict(f'datasets/{args.task}_{config["DATASET_TYPE"]}_{config["DATASET_SIZE"]}.npz')
    dataset = process_dataset(unprocessed_dataset)
    action_dim = dataset["action"].shape[1]
    obs_dim = dataset["obs"].shape[1]
    reward_dim = 1 
    done_dim = 1
    dataset, last_obss, actions, obss, _ = sample_dataset(dataset, args.n_test_data) 
    testing_last_obss = last_obss
    testing_actions = actions
    testing_obss = obss

    # --- train variational models for increasing number of datapoints --- 
    previous_n_train_data = 0         
    for n_train_data in args.n_train_data_list:

        print(f"------ {n_train_data} DATAPOINTS ------")
        base_path_n = f"{base_path}/N{n_train_data}"
        os.makedirs(base_path_n, exist_ok=True)

        # --- augment dataset with newly sampled datapoints --- 
        dataset, last_obss, actions, obss, _ = sample_dataset(dataset, num_samples=n_train_data-previous_n_train_data)      
        if 'training_last_obss' in locals(): 
            training_last_obss = np.vstack((training_last_obss, last_obss))
            training_actions = np.vstack((training_actions, actions))
            training_obss = np.vstack((training_obss, obss))
        else: 
            training_last_obss = last_obss
            training_actions = actions
            training_obss = obss

        # --- normalise data based on current training samples --- 
        normalise_observations, _, _, _ = create_gaussian_normalisation_functions(samples=training_obss)
        normalise_actions, _, _, _ = create_gaussian_normalisation_functions(samples=training_actions)
        normalised_training_inputs = jnp.array(np.concatenate((normalise_observations(training_last_obss), 
                                                                normalise_actions(training_actions)), 
                                                axis=1))
        normalised_training_targets = jnp.array(normalise_observations(training_obss))
        normalised_testing_inputs = jnp.array(np.concatenate((normalise_observations(testing_last_obss), 
                                                             normalise_actions(testing_actions)), 
                                            axis=1))
        normalised_testing_targets = jnp.array(normalise_observations(testing_obss))

        # --- initialise variational mlp --- 
        variational_mlp = VariationalMLP(input_dim = obs_dim + action_dim,
                                         output_dim = obs_dim,
                                         hidden_layers = config["MODEL"]["HIDDEN_LAYERS"],
                                         learn_sigma = config["VARIATIONAL_INFERENCE"]["LEARN_SIGMA"])
        
        # --- train variational mlp --- 
        rng, _rng = jax.random.split(rng)
        eval_history, variational_mlp_params, _, _ = train_variational_mlp(config["VARIATIONAL_INFERENCE"], 
                                                                                (normalised_training_inputs, normalised_training_targets), 
                                                                                (normalised_testing_inputs, normalised_testing_targets), 
                                                                                variational_mlp, 
                                                                                _rng)
        if config["LOG"] == True:
            save_history(eval_history, f"{base_path_n}/eval_history.json")
            mean_tree, cov_tree = log_posterior_distribution(variational_mlp_params, base_path_n, eval_history)
        else:
            mean_tree, cov_tree = log_posterior_distribution(variational_mlp_params)

        previous_n_train_data = n_train_data

    # --- plot 0: visualise training history --- 
        # a: train KL Loss; 
        # b: validation loss; 
        # c: posterior variance; 
        # d: validation MSE --- all v gradient update step
    train_cmap = cm.Purples
    norm = mcolors.LogNorm(vmin=1, vmax=args.n_train_data_list[-1])
    sm = cm.ScalarMappable(cmap=train_cmap, norm=norm) 
    plt.tight_layout()
    fig, axes = plt.subplots(1, 4, figsize=(28, 12))
    vi_val_mse = []
    vi_post_var = []
    for n_train_data in args.n_train_data_list:
        base_path_n = f"{base_path}/N{n_train_data}"
        color_intensity = (n_train_data - args.n_train_data_list[0]) / (args.n_train_data_list[-1] - args.n_train_data_list[0]) if args.n_train_data_list[0] != args.n_train_data_list[-1] else 0.5
        colour = train_cmap(color_intensity) 
        state_log_dict = load_history(f"{base_path_n}/eval_history.json")
        state_epoch = state_log_dict['epoch']
        state_train_loss_kl = state_log_dict['mean_train_kl_loss']
        state_val_loss = state_log_dict['mean_val_loss']
        state_post_var = state_log_dict['post_var']
        state_val_mse = state_log_dict['mean_val_mse_loss']
        vi_post_var.append(state_post_var[-1])
        vi_val_mse.append(state_val_mse[-1])
        axes[0].plot(state_epoch * n_train_data / config["VARIATIONAL_INFERENCE"]["BATCH_SIZE"], state_train_loss_kl, '-o', linewidth=2, color=colour, label=f'{n_train_data}')
        axes[1].plot(state_epoch * n_train_data / config["VARIATIONAL_INFERENCE"]["BATCH_SIZE"], state_val_loss, '-o', linewidth=2, color=colour, label=f'{n_train_data}')
        axes[2].plot(state_epoch * n_train_data / config["VARIATIONAL_INFERENCE"]["BATCH_SIZE"], state_post_var, '-o', linewidth=2, color=colour, label=f'{n_train_data}')
        axes[3].plot(state_epoch * n_train_data / config["VARIATIONAL_INFERENCE"]["BATCH_SIZE"], state_val_mse, '-o', linewidth=2, color=colour, label=f'{n_train_data}')
    axes[0].set_title('KL loss per datapoint')
    axes[0].set_xlabel('Training gradient update steps')
    axes[0].set_ylabel('KL loss')
    axes[0].grid(True)
    axes[1].set_title('Validation loss per datapoint')
    axes[1].set_xlabel('Training gradient update steps')
    axes[1].set_ylabel('Validation loss')
    axes[1].grid(True)
    axes[2].set_title('Posterior variance')
    axes[2].set_xlabel('Training gradient update steps')
    axes[2].set_ylabel('Posterior variance')
    axes[2].grid(True)
    axes[3].set_title('Validation MSE')
    axes[3].set_xlabel('Training gradient update steps')
    axes[3].set_ylabel('Validation MSE')
    axes[3].set_ylim(0, 0.01)
    axes[3].grid(True)
    cbar = fig.colorbar(sm, ax=axes[3], orientation='vertical')
    cbar.set_label('Number of Samples (Log Scale)', fontsize=12)
    plt.savefig(f"{base_path}/vi_training_history.png")
    plt.close()
            
    # --- plot 1: visualise validation MSE v N --- 
    plt.figure(figsize=(14, 6))        
    plt.plot(args.n_train_data_list, vi_val_mse, 'b-', linewidth=2)
    plt.title('MSE on validation set')
    plt.xlabel('N')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.savefig(f"{base_path}/vi_val_mse.png")
    plt.close()
                
    # --- plot 2: visualise final posterior variance v N --- 
    plt.figure(figsize=(14, 6))
    plt.plot(args.n_train_data_list, vi_post_var, 'b-', linewidth=2)
    plt.title('Posterior variance')
    plt.xlabel('N')
    plt.ylabel('Posterior variance')
    plt.grid(True)
    plt.savefig(f"{base_path}/vi_post_var.png")
    plt.close()
    return


if __name__ == "__main__":
    args = get_args()
    learn_posterior_over_state_transition_model_parameters(args)