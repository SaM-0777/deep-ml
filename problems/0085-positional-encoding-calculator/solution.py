import numpy as np

def pos_encoding(position: int, d_model: int):
	# Your code here
	if position == 0 or d_model <= 0:
		return -1

	pe = []

	for pos in range(position):
		sin_cos = []
		for i in range(d_model // 2):
			angle = pos / np.power(10000, 2 * i / d_model)
			sin = np.sin(angle)
			cos = np.cos(angle)
			
			sin_cos.append(sin)
			sin_cos.append(cos)

		pe.append(sin_cos)
	
	return np.asarray(pe, dtype=np.float16)