import numpy as np

def sample_dataset(dataset: dict, 
                   num_samples: int = 1) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    --- function to sample dataset without replacement --- 
    args: 
        dataset: dictionary containing the dataset
        num_samples: number of samples to sample
    returns: 
        dataset: dictionary containing the remaining dataset
        last_obss: numpy array containing the last observations
        actions: numpy array containing the actions
        obss: numpy array containing the observations
        rewards: numpy array containing the rewards
    """
    # --- check dataset length meets sample size criteria --- 
    dataset_length = dataset["obs"].shape[0]
    if num_samples > dataset_length:
        raise ValueError(f"Number of samples ({num_samples}) requested exceeds the remaining dataset size ({dataset_length}).")

    sample_indices = np.random.choice(dataset_length, num_samples, replace=False)

    last_obss = np.array(dataset["last_obs"][sample_indices])
    actions = np.array(dataset["action"][sample_indices])
    obss = np.array(dataset["obs"][sample_indices])
    rewards = np.array(dataset["reward"][sample_indices])

    if actions.ndim == 1:
        actions = np.expand_dims(actions, axis=1)

    if rewards.ndim == 1:
        rewards = np.expand_dims(rewards, axis=1)

    dataset["last_obs"] = np.delete(dataset["last_obs"], sample_indices, axis=0)
    dataset["action"] = np.delete(dataset["action"], sample_indices, axis=0)
    dataset["obs"] = np.delete(dataset["obs"], sample_indices, axis=0)
    dataset["reward"] = np.delete(dataset["reward"], sample_indices, axis=0)

    return dataset, last_obss, actions, obss, rewards