import numpy as np
from typing import Tuple

def softmax(x):
    x = x - np.max(x, axis=-1,keepdims=True)
    x_exp = np.exp(x)
    total_x_exp = np.sum(x_exp, axis=-1, keepdims=True)
    return x_exp / total_x_exp

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    return X @ W_q, X @ W_k, X @ W_v

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    seqlen, d = K.shape
    scores = (Q @ K.T) / np.sqrt(d)
    scores = softmax(scores) @ V
    return scores

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seqlen, d_model = K.shape
    head_dim = d_model // n_heads
    Q_heads = Q.reshape(seqlen, n_heads, head_dim).transpose(1, 0, 2)
    K_heads = K.reshape(seqlen, n_heads, head_dim).transpose(1, 0, 2)
    V_heads = V.reshape(seqlen, n_heads, head_dim).transpose(1, 0, 2)

    outputs = []
    for head in range(n_heads):
        score = self_attention(Q_heads[head], K_heads[head], V_heads[head])
        outputs.append(score)
    
    return np.concatenate(outputs, axis=-1)
