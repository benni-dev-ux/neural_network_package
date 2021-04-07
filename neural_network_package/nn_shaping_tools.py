import numpy as np
import pandas as pd


def add_bias_column(x):
    """
    Adds Bias of 1 as first Column of Array
    """
    return np.c_[np.ones(len(x)), x]


def concatenate_frames(X, frame_amount):
    """
    Concatenates next frames to row of current frame

    """
    combined_frames = X.loc[0:]
    for i in range(1, frame_amount):
        additional_frame = X.loc[i:].reset_index(drop=True)
        combined_frames = pd.concat(
            [combined_frames, additional_frame], axis=1)

    # Delete Last n Rows
    combined_frames.drop(combined_frames.tail(
        frame_amount).index, inplace=True)
    # Convert to numpy array
    return combined_frames.values


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
