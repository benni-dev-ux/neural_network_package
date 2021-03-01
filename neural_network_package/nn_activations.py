import numpy as np


def sigmoid(z):
    """Returns sigmoid of given z"""
    return 1 / (1 + np.e ** -z)
