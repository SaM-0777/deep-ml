import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    # Your code here
    B, C, H, W = X.shape

    channels_per_group = C // num_groups

    x_grouped = X.reshape(B, num_groups, channels_per_group, H, W)
    mean = np.mean(x_grouped, axis=(2,3,4), keepdims=True)
    var = np.var(
        x_grouped,
        axis=(2,3,4),
        keepdims=True
    )

    x_norm =( x_grouped - mean) / np.sqrt(var + epsilon)

    Y = gamma * x_norm.reshape(B, C, H, W) + beta
    return Y
