import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	weights = np.asarray(weights)
	bias = np.asarray(bias)
	activations = []
	se = []
	for i in range(len(features)):
		feature = np.asarray(features[i])
		label = np.asarray(labels[i])
		p = feature @ weights + bias
		act = np.round(1. / (1. + np.exp(-p)), 4)
		activations.append(act)
		se.append((act - label) ** 2)
	
	se = np.asarray(se)
	return (activations, float(se.mean()))
