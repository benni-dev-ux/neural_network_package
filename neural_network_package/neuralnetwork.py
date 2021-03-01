import numpy as np
from tqdm import tqdm

from .nn_feature_scaling import *
from .nn_shaping_tools import *
from .nn_evaluation import *
from .nn_cost_functions import *
from .nn_activations import *


class NeuralNetwork(object):
    '''
    Neural Network
    '''

    def __init__(self, shape):
        self.shape = shape
        self.thetas = initialize_random_thetas(shape)

    def forward_prop(self, x):
        """
        Calculates the Activation for the different Layers of the NN

        Parameters:
        -----------
        X: Input layer
        """
        activations = []
        prev_activation = x
        for t in range(len(self.thetas) - 1):
            a = sigmoid(prev_activation @ self.thetas[t])
            # insert bias at first position of array
            prev_activation = np.c_[np.ones(len(a)), a]
            activations.append(prev_activation)

        # Output layer without bias
        prev_activation = sigmoid(prev_activation @ self.thetas[-1])
        activations.append(prev_activation)

        # return activations in reverse order, so output is at 0
        return activations[::-1]

    def back_prop(self, x, y, thetas, activations, soft_activation):
        """
        Backwards propagates the NN
        Parameters:
        -----------
        X: Input layer
        Y: Outputs
        activations: Activations from forward Prop
        soft_activation: softmax of output layer
        """
        activations.append(x)
        gradient_list = []
        num_samples = len(x)

        # output layer
        sig_derivative = activations[0] * (1 - activations[0])

        delta = (soft_activation - y) * sig_derivative

        gradient = activations[1].T @ delta / num_samples
        gradient_list.append(gradient)
        # hidden layers
        for layer in range(1, len(thetas), 1):
            # current activations without bias column
            curr_activation = activations[layer][:, 1:]
            # previous activations with bias column
            prev_activation = activations[layer + 1].T
            # current theta without bias column
            curr_theta = thetas[-layer][1:].T
            sig_derivative = (curr_activation * (1 - curr_activation))

            delta = delta @ curr_theta * sig_derivative

            gradient = prev_activation @ delta / num_samples
            gradient_list.append(gradient)

        return gradient_list

    def train(self, x, y, alpha, iterations):
        """
        Trains the NN through backpropagation
        X: Input layer
        Y: Outputs
        alpha: learning rate
        iterations: iterations

        returns: trained thetas

        """

        trained_thetas = self.thetas.copy()

        error_history = []
        accuracy_history = []

        for _ in tqdm(range(iterations)):
            activations = self.forward_prop(x)
            soft_activation_output = softmax(activations[0])
            accuracy_history.append(
                accuracy_multiclass(soft_activation_output, y))

            error = categorical_cross_entropy(soft_activation_output, y).mean()
            error_history.append(error)

            gradients = self.back_prop(x, y, trained_thetas,
                                       activations, soft_activation_output)

            # update thetas
            for idx in range(len(trained_thetas)):
                trained_thetas[-(idx + 1)] -= alpha * gradients[idx]

        return error_history, accuracy_history, trained_thetas, soft_activation_output
