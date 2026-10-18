import numpy as np

def newtonschulz5(G: np.ndarray, steps=5, eps=1e-7) -> np.ndarray:
    """
    Apply the Newton-Schulz (quintic) iteration for 5 steps to matrix G.
    Args:
        G: 2D NumPy array
        steps: Number of iteration steps (default 5)
        eps: Small constant for stability
    Returns:
        Matrix after Newton-Schulz iteration
    """
    # Your code here
    row, col = G.shape
    G = G.T if row > col else G
    Gf = np.linalg.norm(G, ord='fro')

    a = 3.4445
    b = -4.7750
    c = 2.0315

    X = G / (Gf + eps)
    for i in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    
    return X.T if row > col else X

def muon_update(theta: np.ndarray, grad: np.ndarray, B_prev: np.ndarray, mu: float, lr: float) -> tuple:
    """
    Performs one Muon optimizer update (Algorithm 2). Returns the updated parameter, new B, and the preconditioned update.
    Args:
        theta: Parameter matrix (2D NumPy array)
        grad: Gradient matrix (same shape)
        B_prev: Previous B matrix (momentum)
        mu: Momentum factor (0 <= mu < 1)
        lr: Learning rate (step size)
    Returns:
        theta_new: Updated parameter matrix
        B_new: Updated B matrix
        O: Preconditioned update
    """
    # Step 1: Compute B_t
    # Step 2: Precondition with Newton-Schulz5
    # Step 3: Update theta
    # Your code here
    B_t = mu * B_prev + grad
    O_t = newtonschulz5(B_t)
    theta -= lr * O_t
    return theta, B_t, O_t