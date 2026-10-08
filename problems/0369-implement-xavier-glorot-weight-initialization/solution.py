import numpy as np

def xavier_init(fan_in: int, fan_out: int, mode: str = 'uniform', seed: int = 42) -> dict:
    """
    Perform Xavier/Glorot weight initialization.

    Args:
        fan_in (int): Number of input units.
        fan_out (int): Number of output units.
        mode (str): 'uniform' or 'normal'.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Contains 'weights' (nested list), 'shape' (list), and 'param' (float).
    """
    # Your code here
    rng = np.random.default_rng(seed)
    if mode == "normal":
        std = np.sqrt(2 / (fan_in + fan_out))
        weights = rng.normal(loc=0, scale=std, size=(fan_in, fan_out))
        param = float(std)
    elif mode == "uniform":
        a = np.sqrt(6 / (fan_in + fan_out))
        weights = rng.uniform(low=-a, high=a, size=(fan_in, fan_out))
        param = float(a)
    else:
        raise ValueError("mode must be 'uniform' or 'normal'")
    
    return {
        "weights": weights,
        "param": param,
        "shape": [fan_in, fan_out],
    }

