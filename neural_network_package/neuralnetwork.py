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

    def train(self, x, y, alpha, batch_size=32, iterations=100, lamda_value=0, beta_val=0):
        """
        Trains the NN through backpropagation
        X: Input layer
        Y: Outputs
        alpha: learning rate
        iterations: iterations
        batch: chunks of data going through
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

        mini_batches = self._create_batches(x, y, batch_size) #creating batches
        print("data shape:", mini_batches.shape)

        for _ in tqdm(range(iterations)):

            activations = None #saving activations to access after each iteration for every batch
            accuracy_for_batch = []
            error_for_batch = []

            np.random.shuffle(mini_batches)
            for mini_batch in mini_batches:
                x_batch = mini_batch[:, :9] #slice only columns (first 9) of one hot vector
                y_batch = mini_batch[:, 9:] #slice starting at 10th
                activations, z_list = self.forward_prop(x_batch)
                soft_activation_output = softmax(activations[0])
                accuracy_for_batch.append(accuracy_multiclass(soft_activation_output, y_batch))

                error = categorical_cross_entropy(soft_activation_output, y_batch).mean()
                error_for_batch.append(error)

                gradients = self.back_prop(x_batch, y_batch, trained_thetas,
                                               activations, soft_activation_output, z_list)

                # update thetas
                for idx in range(len(trained_thetas)):
                    velocity_list[-(idx + 1)] = alpha * (gradients[idx] +
                                                     lamda_value / num_samples * trained_thetas[-(idx + 1)]) + \
                                            beta_val * velocity_list[-(idx + 1)]
                    trained_thetas[-(idx + 1)] -= velocity_list[-(idx + 1)]

            accuracy_history.append(np.mean(accuracy_for_batch))
            error_history.append(np.mean(error_for_batch))

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

    @staticmethod
    def _create_batches(x, y, batch_size):  # private method to create batches
        data = np.hstack((x, y))
        batches = []

        data_l = len(data)
        for index in range(0, data_l, batch_size):
            batches.append(data[index:min(index + batch_size, data_l)]) #create batches with certain size, if not enough data available for last batch create batch with less data

        return np.array(batches)
