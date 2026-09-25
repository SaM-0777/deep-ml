import math
def elu(x: float, alpha: float = 1.0) -> float:
	"""
	Compute the ELU activation function.

	Args:
		x (float): Input value
		alpha (float): ELU parameter for negative values (default: 1.0)

	Returns:
		float: ELU activation value
	"""
	# Your code here
	if x >= 0:
		return x
	return round(alpha * (math.exp(x) - 1),4)