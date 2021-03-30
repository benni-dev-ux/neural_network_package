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

    def __init__(self, shape, thetas=None):
        self.shape = shape
        self.activation_function_name = 'sigmoid'

        # Create random thetas if there are none
        if thetas is None:
            self.thetas = initialize_random_thetas(shape)
        else:
            self.thetas = thetas

    def set_activation_function(self, activation_function_name):
        """
        set activation function to a specific function, otherwise sigmoid

        Parameters:
        -----------
        activation_function_name: 'sigmoid', 'relu' or 'tanh'
        """
        self.activation_function_name = activation_function_name

    def forward_prop(self, x):
        """
        Calculates the Activation for the different Layers of the NN

        Parameters:
        -----------
        X: Input layer

        returns activations list
        """
        z_list = []
        activations = []
        prev_activation = x
        for t in range(len(self.thetas) - 1):
            z = prev_activation @ self.thetas[t]
            a = ACTIVATION_FUNCTION[self.activation_function_name](z)
            # insert bias at first position of array
            prev_activation = np.c_[np.ones(len(a)), a]
            z_list.append(z)
            activations.append(prev_activation)

        # Output layer without bias
        prev_activation = prev_activation @ self.thetas[-1]
        activations.append(prev_activation)
        z_list.append(prev_activation)

        # return activations in reverse order, so output is at 0
        return activations[::-1], z_list[::-1]

    def back_prop(self, x, y, thetas, activations, soft_activation, z_list):
        """
        Backwards propagates the NN
        Parameters:
        -----------
        X: Input layer
        Y: Outputs
        activations: Activations from forward Prop
        soft_activation: softmax of output layer

        returns gradients list
        """
        activations.append(x)
        gradient_list = []
        num_samples = len(x)

        # output layer
        delta = (soft_activation - y)

        gradient = activations[1].T @ delta / num_samples
        gradient_list.append(gradient)
        # hidden layers
        for layer in range(1, len(thetas), 1):
            # current activations/ z without bias column
            curr_activation = activations[layer][:, 1:]
            curr_z = z_list[layer]
            # previous activations with bias column
            prev_activation = activations[layer + 1].T
            # current theta without bias column
            curr_theta = thetas[-layer][1:].T

            activation_derivate = ACTIVATION_FUNCTION_DERIV[self.activation_function_name](curr_z)

            delta = delta @ curr_theta * activation_derivate

            gradient = prev_activation @ delta / num_samples
            gradient_list.append(gradient)

        return gradient_list

    def train(self, x, y, alpha, iterations, lamda_value=0, beta_val=0):
        """
        Trains the NN through backpropagation
        X: Input layer
        Y: Outputs
        alpha: learning rate
        iterations: iterations
        lambda_value: optional value, adds regularization
        beta_value: optional value, adds momentum, should be between 0.5 and 0.99

        returns: error_history, accuracy_history, trained_thetas, soft_activation_output

        """

        trained_thetas = self.thetas.copy()

        error_history = []
        accuracy_history = []
        num_samples = len(x)

        # initialize velocity for momentum
        velocity_list = self.thetas.copy()
        for idx in range(len(velocity_list)):
            velocity_list[idx] = np.zeros(velocity_list[idx].shape)

        for _ in tqdm(range(iterations)):
            activations, z_list = self.forward_prop(x)
            soft_activation_output = softmax(activations[0])
            accuracy_history.append(
                accuracy_multiclass(soft_activation_output, y))

            error = categorical_cross_entropy(soft_activation_output, y).mean()
            error_history.append(error)

            gradients = self.back_prop(x, y, trained_thetas,
                                       activations, soft_activation_output, z_list)

            # update thetas
            for idx in range(len(trained_thetas)):
                velocity_list[-(idx + 1)] = alpha * (gradients[idx] +
                                                     lamda_value / num_samples * trained_thetas[-(idx + 1)]) + \
                                                     beta_val * velocity_list[-(idx + 1)]
                trained_thetas[-(idx + 1)] -= velocity_list[-(idx + 1)]

        return error_history, accuracy_history, trained_thetas, soft_activation_output

    def predict(self, x):
        """
        Predicts Y from given X and existing thetas
        -----------
        X: Input layer
        returns: softmax of the output layer

        """
        activations, z_list = self.forward_prop(x)
        return softmax(activations[0])
