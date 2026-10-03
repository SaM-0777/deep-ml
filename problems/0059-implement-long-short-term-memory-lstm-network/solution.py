import numpy as np

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))
	
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def forward(self, x, initial_hidden_state, initial_cell_state):
		"""
		Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
		"""
		C = initial_cell_state
		h = initial_hidden_state

		outputs = []

		for xt in x:
			xt = xt.reshape(-1, 1)

			combined = np.concatenate([h, xt], axis=0)

			ft = self.sigmoid(self.Wf @ combined + self.bf)
			it = self.sigmoid(self.Wi @ combined + self.bi)

			C_candidate = np.tanh(self.Wc @ combined + self.bc)

			C = (ft * C) + (it * C_candidate)
			ot = self.sigmoid(self.Wo @ combined + self.bo)
			h = np.tanh(C) * ot

			outputs.append(h)

		return outputs, h, C

