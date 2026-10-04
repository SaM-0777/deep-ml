import numpy as np


def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def softmax(x):
	x = x - np.max(x, axis=-1, keepdims=True)
	total = np.sum(np.exp(x), axis=-1, keepdims=True)
	return np.exp(x) / total

def masked_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
	seqlen, d = K.shape
	scores = (Q @ K.T)/np.sqrt(d)
	scores = scores + mask
	scores = softmax(scores)
	attn = scores @ V
	return attn
