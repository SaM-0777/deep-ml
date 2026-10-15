import math
import numpy as np

def l2_normalize(x, eps = 1e-6):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)

def softplus(x):
    return np.logaddexp(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def rms_norm(x, rms_weight, eps):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return (x / rms) * rms_weight

def gated_deltanet(q, k, v, a, b, g, A_log, rms_weight, eps=1e-6):
    """
    Simplified Gated DeltaNet linear attention forward pass.

    Args:
        q, k, v: lists of shape (T, d)
        a, b: lists of shape (T,) -- pre-activation gate logits
        g: list of shape (T, d) -- output gate logits
        A_log: float -- learned log time-scale
        rms_weight: list of shape (d,) -- RMSNorm scale
        eps: float -- numerical stability constant

    Returns:
        Nested list of shape (T, d), rounded to 4 decimals.
    """
    q = np.asarray(q)
    k = np.asarray(k)
    v = np.asarray(v)
    a = np.asarray(a)
    b = np.asarray(b)
    g = np.asarray(g)
    rms_weight = np.asarray(rms_weight)

    T, d = q.shape

    q_norm = l2_normalize(q)
    k_norm = l2_normalize(k)

    alpha_t = np.exp(-softplus(a) * np.exp(A_log))
    beta_t = sigmoid(b)

    S = np.zeros((d, d))

    outputs = []
    for t in range(T):
        S = alpha_t[t] * S

        prediction = S.T @ k_norm[t]
        delta_t = (v[t] - prediction) * beta_t[t]

        S = S + np.outer(k_norm[t], delta_t)

        o_t = S.T @ q_norm[t]

        outputs.append(o_t)
    
    outputs = np.stack(outputs, axis=0) # T, d

    output_norm = rms_norm(outputs, rms_weight, eps)

    y = output_norm * (g * sigmoid(g))

    return np.round(y, 4)