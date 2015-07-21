import numpy as np


def rmse(predictions, targets):
    """
    Root mean squared error
    """
    return np.sqrt(((predictions - targets) ** 2).mean())
