import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	# Your code here
	y_pred = w @ X.T
	return np.mean((y_pred - y_true) ** 2) + alpha * np.sum(w ** 2)
