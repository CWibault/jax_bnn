import numpy as np

def process_dataset(dataset: dict) -> dict:
    """
    --- function to process a curated dataset and remove all done = true datapoints --- 
    args: dataset: dictionary containing the dataset
    returns: dataset: dictionary containing the processed dataset
    """
    if dataset["action"].ndim == 1:
        dataset["action"] = np.expand_dims(dataset["action"], axis=1)
    if dataset["reward"].ndim == 1:
        dataset["reward"] = np.expand_dims(dataset["reward"], axis=1)
    dataset["last_obs"] = np.delete(dataset["last_obs"], np.where(dataset["done"] == 1), axis=0)
    dataset["action"] = np.delete(dataset["action"], np.where(dataset["done"] == 1), axis=0)
    dataset["obs"] = np.delete(dataset["obs"], np.where(dataset["done"] == 1), axis=0)
    dataset["reward"] = np.delete(dataset["reward"], np.where(dataset["done"] == 1), axis=0)
    del dataset["done"]
    original_dataset_length = dataset["last_obs"].shape[0]
    assert len(dataset["last_obs"]) == len(dataset["action"]) == len(dataset["obs"]) == len(dataset["reward"]), "All datasets must be of the same length"

    return dataset
