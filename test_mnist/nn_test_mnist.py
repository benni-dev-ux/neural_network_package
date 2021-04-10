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
X_validation = nnp.add_bias_column(X_validation_scaled)

# Defining Y Values as One-Hot-Vectors
Y = np.identity(10, dtype=int)[labels_train]
Y_validation = np.identity(10, dtype=int)[labels_validation]

# %% Running the NeuralNet
num_samples = 600
alpha = 10
epoch = 100
batch_size = 10
lamda_value = 0
beta_val = 0

# initialize a new neural net
neural_net = nnp.NeuralNetwork([784, 20, 10])
neural_net.set_activation_function("tanh")

error_history, accuracy_history, gradients, softmax = neural_net.train(x=X[:num_samples], y=Y[:num_samples],
                                                                       alpha=alpha, epoch=epoch)

print("Accuracy Training ", accuracy_history[-1])
print("Error Training ", error_history[-1])
generated_trainig_result = neural_net.predict(X[:num_samples])
print("F1 Training ", nnp.f1_score(generated_trainig_result, Y[:num_samples]))

# %% Validate Training
num_validation_samples = 100

prediction_result = neural_net.predict(X_validation[:num_validation_samples])

print("F1 Validation ", nnp.f1_score(prediction_result, Y_validation[:num_validation_samples]))
