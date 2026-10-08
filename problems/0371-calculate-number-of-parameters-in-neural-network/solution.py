import numpy as np

def calculate_conv2d_param(in_channel, out_channel, kernel_size, bias):
	if bias:
		return (in_channel * kernel_size * kernel_size * out_channel) + out_channel
	else:
		return (in_channel * kernel_size * kernel_size * out_channel)


def calculate_dense_param(input_size, output_size, bias):
	if bias:
		return (input_size * output_size) + output_size
	return (input_size * output_size)

def calculate_parameters(layers: list[dict]) -> int:
	"""
	Calculate the total number of trainable parameters in a neural network.

	Args:
		layers: List of dictionaries, each describing a layer.

	Returns:
		Total number of trainable parameters (int).
	"""
	# Your code here
	total_param = 0
	for layer in layers:
		if layer["type"] == "dense":
			bias = layer.get("bias", True)
			total_param += calculate_dense_param(layer["input_size"], layer["output_size"], bias)
		else:
			bias = layer.get("bias", True)
			total_param += calculate_conv2d_param(layer["in_channels"], layer["out_channels"], layer["kernel_size"],bias)
	return total_param