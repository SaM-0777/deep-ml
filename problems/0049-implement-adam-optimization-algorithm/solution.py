import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.
    # Your code here
    m = 0
    v = 0
    for iteration in range(num_iterations):
        t = iteration + 1
        g = grad(x0)

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)

        m_hat = m / (1 - (beta1 ** t))
        v_hat = v / (1 - (beta2 ** t))

        x0 -= (learning_rate * m_hat) / (np.sqrt(v_hat) + epsilon)
    
    return x0


