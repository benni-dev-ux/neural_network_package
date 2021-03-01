import numpy as np


def add_bias_column(x):
    """
    Adds Bias of 1 as first Column of Array
    """
    return np.c_[np.ones(len(x)), x]


def initialize_random_thetas(shape):
    """
    initializes random set of thetas for a given shape
    range -0.5 to 0.5
    automatically adds bias
    """
    theta_list = []
    np.random.seed(1)
    for i in range(len(shape) - 1):
        theta_list.append(np.random.rand((shape[i]) + 1, shape[i + 1])-0.5)

    return theta_list


