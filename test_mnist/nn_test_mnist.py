# %% import modules
from mnist.loader import MNIST

import neural_network_package as nnp
import mnist_downloader
import numpy as np
import matplotlib.pyplot as plt

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

# Add Bias to X_pizza
X = nnp.add_bias_column(X_train_scaled)
X_validation = nnp.add_bias_column(X_validation_scaled)

# Defining Y_pizza Values as One-Hot-Vectors
Y = np.identity(10, dtype=int)[labels_train]
Y_validation = np.identity(10, dtype=int)[labels_validation]

# %% Running the NeuralNet

# define hyperparameter
alpha = 10
epoch = 5
batch_size = 300
lamda_value = 0.1
beta_val = 0.3

# initialize a new neural net
neural_net = nnp.NeuralNetwork([784, 20, 10])
neural_net.set_activation_function("tanh")

# train
error_history, accuracy_history, gradients, softmax = neural_net.train(x=X, y=Y,
                                                                       alpha=alpha, epochs=epoch, batch_size=batch_size,
                                                                       lamda_value=lamda_value, beta_value=beta_val)
# console output
print("Accuracy Training ", accuracy_history[-1])
print("Error Training ", error_history[-1])
generated_trainig_result = neural_net.predict(X)
print("F1 Training ", nnp.f1_score(generated_trainig_result, Y))

# %% Plot Training Progress
fig, ax = plt.subplots()
ax.plot(error_history, label="Error")
ax.set_xlabel("Iterations")
ax.set_ylabel("Error")
ax.set_title("Error History")
fig.legend()
fig.tight_layout()
plt.show()

# %% Validate Training

prediction_result = neural_net.predict(X_validation)
print("F1 Validation ", nnp.f1_score(prediction_result, Y_validation))
