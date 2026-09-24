import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	m = max(scores)
	scores = np.asarray(scores)
	return scores - (m + np.log(np.sum(np.exp(scores - m))))


