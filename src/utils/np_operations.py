import numpy as np

def silu(x: np.ndarray) -> np.ndarray:
    return x * (1 / (1 + np.exp(-x)))