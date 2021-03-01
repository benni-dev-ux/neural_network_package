# %% import modules
from mnist.loader import MNIST

import neural_network_package as nnp
import mnist_downloader
import numpy as np


# %% Download MNIST Dataset

download_folder = "./mnist/"
mnist_downloader.download_and_unzip(download_folder)

# %% load training data
mndata = MNIST('mnist', return_type="numpy")
images_train, labels_train = mndata.load_training()
images_validation, labels_validation = mndata.load_testing()

# %% Scale Data
scaler = nnp.StandardScaler()
scaler.fit(images_train)
X_train_scaled = scaler.transform(images_train)
X_validation_scaled = scaler.transform(images_validation)

# Add Bias to X
X = nnp.add_bias_column(X_train_scaled)

# Defining Y Values as One-Hot-Vectors
Y = np.identity(10, dtype=int)[labels_train]

# %% Running the NeuralNet
num_samples = 100
alpha = 2
iterations = 100

# initialize a new neural net
neural_net = nnp.NeuralNetwork([784, 20, 10])

error_history, accuracy_history, gradients, softmax = neural_net.train(
    X[:num_samples], Y[:num_samples], alpha, iterations)

print(accuracy_history)
