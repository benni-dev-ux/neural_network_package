import pandas as pd
import neural_network_package as nnp
import numpy as np

# -------------------------------------------------------------------------------------------------
# Linear Regression

# get data
pizza_data = pd.read_csv("./pizzas.csv")
features = pizza_data[["flour", "tomatoes", "mozzarella", "salt", "diameter", "delivery"]].values
X_pizza = nnp.add_bias_column(features)
Y_pizza = pizza_data["price"].values

m_pizza = X_pizza.shape[1]  # returns the number of columns in X_pizza

neural_net_pizza = nnp.NeuralNetwork([m_pizza - 1, 1])
trained_thetas_pizza, error_hist_pizza = neural_net_pizza.train_linear_regression(X_pizza, Y_pizza, 1, 10)

# -------------------------------------------------------------------------------------------------
# Logistic Regression with new functions in network

delivery_data = pd.read_csv("./delivery_data.csv")

features_log = delivery_data[["motivation", "distance"]].values
X_delivery = nnp.add_bias_column(features_log)
Y_delivery = delivery_data["delivery?"].values

m_del = X_delivery.shape[1]
neural_net_delivery = nnp.NeuralNetwork([m_del - 1, 1])
trained_thetas_delivery, error_hist_delivery = neural_net_delivery.train_logistic_regression(X_delivery, Y_delivery, 1,
                                                                                             10)
prediction = nnp.logistic_regression(X_delivery, trained_thetas_delivery)
accuracy_delivery = nnp.accuracy(prediction, Y_delivery)

# -------------------------------------------------------------------------------------------------
# Logistic Regression with neural net

neural_net_delivery_nn = nnp.NeuralNetwork([m_del - 1, 2])
Y_delivery_one_hot = np.identity(2, dtype=int)[Y_delivery]
error_history, accuracy_history, gradients, softmax = neural_net_delivery_nn.train(x=X_delivery, y=Y_delivery_one_hot,
                                                                                   alpha=1, epoch=10, batch_size=300,
                                                                                   lamda_value=0, beta_val=0)
accuracy_delivery_nn = nnp.accuracy_multiclass(softmax, Y_delivery_one_hot)
