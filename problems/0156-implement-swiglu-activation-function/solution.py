import numpy as np

def SiLU(x):
    return x / (1 + np.exp(-x))


def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    # Your code here
    x1, x2 = np.split(x, 2, axis=1)
    scores = x1 * SiLU(x2)
    return scores