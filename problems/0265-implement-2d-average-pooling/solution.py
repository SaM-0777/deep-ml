import numpy as np

def avg_pool_2d(input_matrix: list[list[float]], pool_size: int) -> list[list[float]]:
	"""
	Perform 2D average pooling on the input matrix.
	
	Args:
		input_matrix: 2D input array of shape (H, W)
		pool_size: Size of the square pooling window
		
	Returns:
		2D array after average pooling of shape (H//pool_size, W//pool_size)
	"""
	# Your code here
	input_matrix = np.asarray(input_matrix)
	H, W = input_matrix.shape

	output = input_matrix.reshape(
		H // pool_size,
		pool_size,
		W // pool_size,
		pool_size
	).mean(axis=(1, 3))
	return output.tolist()
