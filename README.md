# Neural Network Package (Group 07)

Neural Network Package for the Machine Learning Course in WS20 Group 07

## Installation Guide

1. Clone Repository
2. Navigate to module 'neural_network_package'
3. Run command to install package locally

   ` pip install .`

4. Test the installation anywhere on your machine by running

   ` import neural_network_package as nnp`

   ` nnp.test()`

## Data Preparation for Machine Learning

### Feature Scaling

1. Create a scaler and fit it according to the training data (x). Two scalers are available:

   - NormalScaler: Rescale values so that each feature's minimum value is 0 and their maximum value is 1
   - StandardScaler: Centers data around zero with standard derivation of 1. It is more robust with outliers

   ` scaler = nnp.StandardScaler()`

   ` scaler.fit(x)`

2. Transform your data using the initialized scaler

   ` x_scaled = scaler.transform(x)`

### Add Bias Column

3. To use this package for machine learning, it is necessary to add a bias column to the training data before training.

   ` X = nnp.add_bias_column(x_scaled)`

## Using Neural Networks

1. Instantiate a new neural network with the wanted shape
   (first position = input layer (number of features), last postion = ouput layer, layers in between = hidden layer).

   ` neural_net = nnp.NeuralNetwork([input_layer_size, hidden_layer_size, output_layer_size])`

2. Set an activation function to a specific function, otherwise sigmoid will be used for training. Valid parameters are 'sigmoid', 'relu' or 'tanh'.
   ` neural_net.set_activation_function( activation_function_name)`

3. Train the neural network with given inputs, ground truth and various hyperparameters.

   hyperparameters:

   - x : dataset as numpy array
   - y : ground truth as one hot vectors
   - alpha : learning rate
   - batch_size : number of training examples present in a single batch
   - epochs : number of iterations over full dataset
   - lambda_value : optional value, adds regularization
   - beta_value: optional value, adds momentum, should be between 0.5 and 0.99

   ` error_history, accuracy_history, trained_thetas, soft_activation_output = neural_net.train(x, y, alpha, batch_size, epochs, lamda_value, beta_val)`

4. Make predictions from the trained network with validation data.

   ` prediction_result = neural_net.predict(x_validation)`

See nn_test_mnist.py for an exemplary usage of the network with the MNIST dataset and one hidden layer with 20 Neurons

## Using Logistic Regression

### With Neural Network

1. Instantiate a new neural network with the desired shape
   (first position = input layer (number of features), last postion = ouput layer).

   ` neural_net = nnp.NeuralNetwork([input_layer_size, output_layer_size = 2])`

2. Do not set another activation function as sigmoid.

3. See [Using Neural Networks](#Using-Neural-Networks) for further steps.

See nn_test_regression.py for an exemplary usage of linear and logistic regression.

### With explicit functions

1. Instantiate a new neural network without hidden layers.

   ` neural_net = nnp.NeuralNetwork([input_layer_size, output_layer_size = 1])`

2. Train the neural network with given inputs, ground truth and various hyperparameters.

   hyperparameters:

   - x : dataset as numpy array
   - y : ground truth as numpy array
   - alpha : learning rate
   - iterations : number of iterations over full dataset

   ` trained_thetas, error_history = neural_net.train_logistic_regression(x, y, alpha, iterations)`

## Using Linear Regression

1. Instantiate a new neural network without hidden layers.

   ` neural_net = nnp.NeuralNetwork([input_layer_size, output_layer_size = 1])`

2. Train the neural network with given inputs, ground truth and various hyperparameters.

   hyperparameters:

   - x : dataset as numpy array
   - y : ground truth as numpy array
   - alpha : learning rate
   - iterations : number of iterations over full dataset

   ` trained_thetas, error_history = neural_net.train_linear_regression(x, y, alpha, iterations)`

## Evaluation

There are several functions that can be used to evaluate the performance of the network.

1. Accuracy for multiclass classification

   parameters:

   - h : prediction results (as softmax)
   - y : ground truth as one hot vectors

   ` accuracy = nnp.accuracy_multiclass(h, y)`

2. F1 Score

   The F1 score calculates the accuracy from the precision and recall of the prediction results.

   parameters:

   - h : prediction results (as softmax)
   - y : ground truth as one hot vectors
   - true_negative_value : value that should be considered as true_negative, for example 0

   ` f1_score = nnp.f1_score(h, y, true_negative_value)`

3. Accuracy for binary classification

   parameters:

   - h : prediction results (as one dimensional numpy array)
   - y : ground truth (as one dimensional numpy array)

   ` accuracy = nnp.accuracy(h, y)`
