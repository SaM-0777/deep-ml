import numpy as np

def GeLU(x: np.ndarray) -> np.ndarray:
	# Your code here
	def phi(x):
		r = np.sqrt(2) / np.sqrt(np.pi)
		return 1 + np.tanh(r * (x + 0.044715 * x ** 3))

	x = np.asarray(x)
	x = 0.5 * x * phi(x)
	
	return x