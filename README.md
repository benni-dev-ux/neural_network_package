# Neural Network Package (Group 07)

Neural Network Package for the Machine Learning Course in WS20 Group 07

Follow Installation Guide and Quick Start Guide for instructions

## Installation Guide

1. Clone Repository
2. Navigate to Module 'neural_network_package'
3. Run command to install package locally

   ` pip install .`

4. Test the installation anywhere on your Machine by running

   ` import neural_network_package as nnp`

   ` nnp.test()`

## Using Neural Networks

1. Instantiate a new Neural Network with the wanted shape
   (First Position = Input Layer (Number of Features), Last Postion = Ouput Layer, Layers in between = Hidden Layer).

   ` neural_net = nnp.NeuralNetwork([input_layer_size, hidden_layer_size, output_layer_size])`

2. Train the neural network with given inputs, ground truth and various hyperparameter.
   
   hyperparameter:
   - x : dataset as numpy array
   - y : ground truth as one hot vectors
   - alpha : learning rate
   - batch_size : number of training examples present in a single batch
   - epochs : number of iterations over full dataset
   - lambda_value : optional value, adds regularization
   - beta_value: optional value, adds momentum, should be between 0.5 and 0.99

   ` error_history, accuracy_history, trained_thetas, soft_activation_output = neural_net.train(x, y, alpha, batch_size, epochs, lamda_value, beta_val)`

1. Make predictions from the trained network with validation data.

   ` prediction_result = neural_net.predict(x_validation)`

See nn_test_mnist.py for an exemplary usage of the Network with the MNIST Dataset and a hidden layer with 20 Neurons

## Using Linear Regression

1. Instantiate a new Neural Network without hidden layers.

   ` neural_net = nnp.NeuralNetwork([input_layer_size, output_layer_size = 1])`

2. Train the neural network with given inputs, ground truth and various hyperparameter.

   hyperparameter:
   - x : dataset as numpy array
   - y : ground truth as numpy array
   - alpha : learning rate
   - iterations : number of iterations over full dataset

   ` trained_thetas, error_history = neural_net.train_linear_regression(x, y, alpha, iterations`

## Using Logistic Regression

### With explicit functions

1. Instantiate a new Neural Network without hidden layers.

   ` neural_net = nnp.NeuralNetwork([input_layer_size, output_layer_size = 1])`

2. Train the neural network with given inputs, ground truth and various hyperparameter.

   hyperparameter:
   - x : dataset as numpy array
   - y : ground truth as numpy array
   - alpha : learning rate
   - iterations : number of iterations over full dataset

   ` trained_thetas, error_history = neural_net.train_logistic_regression(x, y, alpha, iterations`

### With Neural Network

1. Instantiate a new Neural Network with the wanted shape
   (First Position = Input Layer (Number of Features), Last Postion = Ouput Layer).

   ` neural_net = nnp.NeuralNetwork([input_layer_size, output_layer_size = 2])`

2. See [Using Neural Networks](#Using-Neural-Networks) for further steps

See nn_test_regression.py for an exemplary usage of linear and logistic regression

