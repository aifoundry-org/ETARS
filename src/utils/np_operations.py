import numpy as np

def silu(x: np.ndarray) -> np.ndarray:
    return x * (1 / (1 + np.exp(-x)))

def mse_loss(input, target, reduction='mean'):
    """
    Computes the Mean Squared Error loss between input and target NumPy arrays.

    This function mimics the behavior of torch.nn.functional.mse_loss.

    Args:
        input (np.ndarray): The input array.
        target (np.ndarray): The target array, which must have the same shape as the input.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied, and a full-sized loss tensor is returned.
            'mean': the sum of the output will be divided by the number of elements in the output.
            'sum': the output will be summed.
            Default: 'mean'.

    Returns:
        np.ndarray or float: The calculated loss. If reduction is 'none', it returns an array
                             of the same shape as the input. Otherwise, it returns a scalar.
    """
    if target.shape != input.shape:
        raise ValueError(f"Target shape ({target.shape}) must be the same as input shape ({input.shape})")

    # Calculate the squared difference element-wise
    squared_error = np.square(input - target)

    # Apply the specified reduction
    if reduction == 'none':
        return squared_error
    elif reduction == 'mean':
        return np.mean(squared_error)
    elif reduction == 'sum':
        return np.sum(squared_error)
    else:
        raise ValueError(f"Invalid reduction type: {reduction}. Supported types are 'none', 'mean', and 'sum'.")