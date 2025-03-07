import numpy as np
import jax.numpy as jnp

def create_gaussian_normalisation_functions(samples = None, 
                                            constants_1 = None, 
                                            constants_2 = None):
    """
    --- function to create normalisation and denormalisation functions that work with both numpy and jax arrays --- 
    args:
        samples (np.ndarray or jnp.ndarray, optional): an array of shape (n_samples, n_features) containing the dataset. Used to compute mean and std.
        mean (np.ndarray or jnp.ndarray, optional): pre-computed mean values for each dimension
        std (np.ndarray or jnp.ndarray, optional): pre-computed std values for each dimension
    returns: 
        normalise (function): A function to normalise an array using the provided or computed mean and std.
        denormalise (function): A function to denormalise an array using the provided or computed mean and std.
    """
    mean = constants_1
    std = constants_2

    if samples is not None:
        samples_np = np.asarray(samples)
        mean = np.mean(samples_np, axis=0)
        std = np.std(samples_np, axis=0)
    elif mean is None or std is None:
        raise ValueError("Either `samples` or both `mean` and `std` must be provided.")
    
    # --- avoid division by zero --- 
    std = np.where(std == 0, 1, std) 
    
    def normalise(array):
        xp = jnp if isinstance(array, jnp.ndarray) else np  
        return (array - xp.array(mean)) / xp.array(std)
    
    def denormalise(array):
        xp = jnp if isinstance(array, jnp.ndarray) else np      
        return array * xp.array(std) + xp.array(mean)
    
    return normalise, denormalise, mean, std