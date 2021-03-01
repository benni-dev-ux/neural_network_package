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

## Quick Start Guide

1. Define Inputs and Outputs from your Dataset

2. Instantiate a new Neural Network with the wanted shape
   (First Position = Input Layer, Last Postion = Ouput layer)

` neural_net = nnp.NeuralNetwork([784, 20, 10])`

3. Train the neural network with given learning rate and iterations

` neural_net.train(X, Y, alpha, iterations)`

See nn_test_mnist.py for an exemplary usage of the Network with the MNIST Dataset and a hidden layer with 20 Neurons
