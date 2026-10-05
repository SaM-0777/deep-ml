import numpy as np

def softplus(x):
    return np.log(1 + np.exp(x))

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    # Your code here
    B = X.shape[0]
    h = X @ W_g
    sigma = softplus(X @ W_noise)
    h += sigma * N
    top_k_ind = np.argsort(h, axis=1)[:, -k:]
    masked_h = np.full_like(h, -np.inf)
    rows = np.arange(X.shape[0])[:, None]
    masked_h[rows, top_k_ind] = h[rows, top_k_ind]

    return softmax(masked_h)
