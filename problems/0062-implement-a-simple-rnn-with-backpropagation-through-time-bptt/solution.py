
import numpy as np

class SimpleRNN:
	def __init__(self, input_size, hidden_size, output_size):
		self.hidden_size = hidden_size
		self.W_xh = np.random.randn(hidden_size, input_size)*0.01
		self.W_hh = np.random.randn(hidden_size, hidden_size)*0.01
		self.W_hy = np.random.randn(output_size, hidden_size)*0.01
		self.b_h = np.zeros((hidden_size, 1))
		self.b_y = np.zeros((output_size, 1))

		self.h = None
		self.o = None

	def mse(self, label, y):
		return 0.5 * np.sum((label - y) ** 2)

	def forward(self, x):
		ht = np.zeros((self.hidden_size, 1))
		
		hidden_states = []
		outputs = []
		
		for x_t in x:
			x_t = x_t.reshape(-1, 1)
			ht = np.tanh(self.W_xh @ x_t + self.W_hh @ ht + self.b_h)
			yt = self.W_hy @ ht + self.b_y

			hidden_states.append(ht.copy())
			outputs.append(yt.copy())
		
		self.h = np.stack(hidden_states)
		self.o = np.stack(outputs)

		return self.o


	def backward(self, x, y, learning_rate):
		dW_xh = np.zeros_like(self.W_xh)
		dW_hh = np.zeros_like(self.W_hh)
		dW_hy = np.zeros_like(self.W_hy)

		db_h = np.zeros_like(self.b_h)
		db_y = np.zeros_like(self.b_y)

		dh_next = np.zeros((self.hidden_size, 1))

		for t in reversed(range(len(x))):
			x_t = x[t].reshape(-1, 1)
			y_t = y[t].reshape(-1, 1)

			h_t = self.h[t]
			o_t = self.o[t]

			if t > 0:
				h_t_1 = self.h[t - 1]
			else:
				h_t_1 = np.zeros((self.hidden_size, 1))
			
			dldy = o_t - y_t
			dW_hy += dldy @ h_t.T
			db_y += dldy

			dh = self.W_hy.T @ dldy
			dh += dh_next

			da = dh * (1 - h_t ** 2)
			dW_xh += da @ x_t.T
			dW_hh += da @ h_t_1.T
			db_h += da
			dh_next = self.W_hh.T @ da
		
		self.W_xh -= learning_rate * dW_xh
		self.W_hh -= learning_rate * dW_hh
		self.W_hy -= learning_rate * dW_hy

		self.b_h -= learning_rate * db_h
		self.b_y -= learning_rate * db_y
