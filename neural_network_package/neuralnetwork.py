from tqdm import tqdm

from .nn_activations import *
from .nn_cost_functions import *
from .nn_evaluation import *
from .nn_shaping_tools import *



class NeuralNetwork(object):
    """
    Neural Network
    """

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
        X_pizza: Input layer

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
        X_pizza: Input layer
        Y_pizza: Outputs
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

            activation_derivate = ACTIVATION_FUNCTION_DERIV[self.activation_function_name](
                curr_z)

            delta = delta @ curr_theta * activation_derivate

            gradient = prev_activation @ delta / num_samples
            gradient_list.append(gradient)

        return gradient_list

    def train(self, x, y, alpha, batch_size, epochs, lamda_value=0, beta_value=0):
        """
        Trains the NN through backpropagation
        x : dataset as numpy array
        y : ground truth as one hot vectors
        alpha: learning rate
        batch_size: number of training examples present in a single batch
        epochs: number of iterations over full dataset
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

        num_features = x.shape[1]
        mini_batches = self._create_batches(
            x, y, batch_size)  # creating batches

        for _ in tqdm(range(epochs)):

            accuracy_for_batch = []
            error_for_batch = []

            # np.random.shuffle(mini_batches)
            for mini_batch in mini_batches:
                x_batch = mini_batch[:, :num_features]
                y_batch = mini_batch[:, num_features:]
                activations, z_list = self.forward_prop(x_batch)
                soft_activation_output = softmax(activations[0])
                accuracy_for_batch.append(accuracy_multiclass(
                    soft_activation_output, y_batch))

                error = categorical_cross_entropy(
                    soft_activation_output, y_batch).mean()
                error_for_batch.append(error)

                gradients = self.back_prop(x_batch, y_batch, trained_thetas,
                                           activations, soft_activation_output, z_list)

                # update thetas
                for idx in range(len(trained_thetas)):
                    velocity_list[-(idx + 1)] = alpha * (gradients[idx] +
                                                         lamda_value / num_samples * trained_thetas[-(idx + 1)]) + \
                                                beta_value * velocity_list[-(idx + 1)]
                    trained_thetas[-(idx + 1)] -= velocity_list[-(idx + 1)]

            accuracy_history.append(np.mean(accuracy_for_batch))
            error_history.append(np.mean(error_for_batch))

        return error_history, accuracy_history, trained_thetas, soft_activation_output

    def predict(self, x):
        """
        Predicts y from given x and existing thetas
        -----------
        x: Input layer
        returns: softmax of the output layer

        """
        activations, z_list = self.forward_prop(x)
        return softmax(activations[0])

    @staticmethod
    def _create_batches(x, y, batch_size):
        """
        splits data into batches of certain size

        x: data
        y: ground_truth of data
        batch_size: batch size
        """
        data = np.hstack((x, y))
        batches = []

        data_l = len(data)
        for index in range(0, data_l, batch_size):
            # create batches with certain size, if not enough data available for last batch create batch with less data
            batches.append(data[index:min(index + batch_size, data_l)])

        return np.array(batches)

    def train_linear_regression(self, x, y, alpha, iterations):
        """
        uses a linear regression to train thetas
        x : dataset as numpy array
        y : ground truth as numpy array
        alpha : learning rate
        iterations : number of iterations over full dataset
        """
        optimized_thetas = self.thetas[0].flatten().copy()
        error_history = []
        for i in range(iterations):
            h = linear_regression(x, optimized_thetas)
            error = mse(h, y)
            error_history.append(error)
            optimized_thetas -= alpha * linear_logistic_regression_derivative(h, y, x)

        return optimized_thetas, error_history

    def train_logistic_regression(self, x, y, alpha, iterations):
        """
        uses a logistic regression to train thetas
        x: Input layer
        y: Outputs
        alpha: learning rate
        iterations: iterations
        """
        trained_thetas = self.thetas[0].flatten().copy()
        error_history = []
        for i in range(iterations):
            h = logistic_regression(x, trained_thetas)
            error = cross_entropy(h, y)
            error_history.append(error)
            trained_thetas -= alpha * linear_logistic_regression_derivative(h, y, x)

        return trained_thetas, error_history


def linear_regression(x, thetas):
    """
    calculates linear regression
    """
    return x @ thetas


def logistic_regression(X, thetas):
    """
    calculates logistic regression
    """
    return sigmoid(linear_regression(X, thetas))


def linear_logistic_regression_derivative(h, y, x):
    """
    calculates derivation for linear and logistic regression
    """
    m = len(x)
    return (1 / m) * ((h - y) @ x)
