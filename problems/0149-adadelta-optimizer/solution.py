import numpy as np

def adadelta_optimizer(parameter, grad, u, v, rho=0.95, epsilon=1e-6):
    """
    Update parameters using the AdaDelta optimizer.
    AdaDelta is an extension of AdaGrad that seeks to reduce its aggressive,
    monotonically decreasing learning rate.
    Args:
        parameter: Current parameter value
        grad: Current gradient
        u: Running average of squared gradients
        v: Running average of squared parameter updates
        rho: Decay rate for the moving average (default=0.95)
        epsilon: Small constant for numerical stability (default=1e-6)
    Returns:
        tuple: (updated_parameter, updated_u, updated_v)
    """
    # 1. Input Validation: Ensure arrays are numpy objects and handle scalars cleanly
    parameter = np.asarray(parameter, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    # Verify shape matching
    if not (parameter.shape == grad.shape == u.shape == v.shape):
        raise ValueError("Shapes of parameter, grad, u, and v must all match.")
        
    # Verify hyperparameter bounds
    if not (0 <= rho <= 1):
        raise ValueError("Decay rate rho must be between 0 and 1.")
    if epsilon <= 0:
        raise ValueError("Epsilon must be a strictly positive non-zero value.")

    # 2. Accumulate Gradient (Moving average of squared gradients)
    updated_u = rho * u + (1 - rho) * (grad ** 2)
    
    # 3. Compute the Parameter Update (delta)
    rms_grad = np.sqrt(updated_u + epsilon)
    rms_delta = np.sqrt(v + epsilon)  # uses previous step v_{t-1}
    
    delta = - (rms_delta / rms_grad) * grad
    
    # 4. Accumulate Updates (Moving average of squared parameter updates)
    updated_v = rho * v + (1 - rho) * (delta ** 2)
    
    # 5. Apply the Update
    updated_parameter = parameter + delta
    
    # Return as original shape type (e.g., if input was scalar, return scalar)
    if updated_parameter.ndim == 0:
        return float(updated_parameter), float(updated_u), float(updated_v)
        
    return updated_parameter, updated_u, updated_v