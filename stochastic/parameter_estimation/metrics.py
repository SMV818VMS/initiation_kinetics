import numpy as np
from scipy.stats import pearsonr
"""
Just include normalized distance metrics
"""


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, np.sqrt(variance))


def euclid(prediction, target):

    return np.linalg.norm(prediction - target)


def rmse(prediction, target):
    """
    Root mean squared error
    """
    return np.sqrt(np.mean((prediction - target) ** 2))


def mean_ae(prediction, target):
    """
    Median absolute error
    """
    return np.mean(np.abs(prediction - target))


def median_ae(prediction, target):
    """
    Mean absolute error
    """
    return np.median(np.abs(prediction - target))


def pearson(prediction, target):
    """
    Pearson correlation coefficient
    """
    return pearsonr(prediction, target)[0]


def mse(prediction, target):
    """
    mean squared error
    """
    return np.mean((prediction - target) ** 2)
