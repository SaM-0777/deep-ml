import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	# 1
	input_sequence = np.asarray(input_sequence)
	h = np.asarray(initial_hidden_state)
	Wx = np.asarray(Wx)
	Wh = np.asarray(Wh)
	b = np.asarray(b)

	for x_t in input_sequence:
		h = np.tanh(x_t @ Wx.T + h @ Wh.T + b)

	return np.round(h, 4)




