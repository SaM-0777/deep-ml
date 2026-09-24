import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	weights = initial_weights.copy()
	bias = initial_bias
	mse_values = []
	
	def forward(inp):
		z = inp @ weights + bias
		pred = 1 / (1 + np.exp(-z))
		return pred
	
	def loss_fn(pred, label):
		return (pred - label) ** 2
	
	def compute_mse(preds, ys):
		return np.mean((preds - ys) ** 2)
	
	def compute_grads(preds, inps, labels):
		dz = 2 * (preds - labels) * preds * (1 - preds)
		dw = np.mean(dz[:, None] * inps, axis=0)
		db = np.mean(dz)
		return dw, db
	
	for epoch in range(epochs):
		inps = []
		preds = []
		ys = []
		for i in range(len(features)):
			feature = features[i]
			label = labels[i]
			pred = forward(feature)
			
			preds.append(pred)
			ys.append(label)
			inps.append(feature)
		
		preds = np.asarray(preds)
		ys = np.asarray(ys)
		inps = np.asarray(inps)
		
		# MSE BEFORE parameter update
		mse = compute_mse(preds, ys)
		mse_values.append(float(np.round(mse, 4)))
		
		# Batch gradients
		dw, db = compute_grads(preds, inps, ys)
		
		# ONE update per epoch
		weights -= learning_rate * dw
		bias -= learning_rate * db
	
	return weights, bias, mse_values