import pandas as pd
import neural_network_package as nnp

# %%
# Linear Regression

# get data
pizza_data = pd.read_csv("./pizzas.csv")
features = pizza_data[["flour", "tomatoes", "mozzarella", "salt", "diameter", "delivery"]].values
X_pizza = nnp.add_bias_column(features)
Y_pizza = pizza_data["price"].values

m = X_pizza.shape[1]  # returns the number of columns in X_pizza

neural_net_pizza = nnp.NeuralNetwork([0, m])
trained_thetas_pizza, error_hist_pizza = neural_net_pizza.train_linear_regression(X_pizza, Y_pizza, 1, 10)

# %%-------------------------------------------------------------------------------------------------
# Logistic Regression

delivery_data = pd.read_csv("./delivery_data.csv")

features_log = delivery_data[["motivation", "distance"]].values
X_delivery = nnp.add_bias_column(features_log)
Y_delivery = delivery_data["delivery?"]

m = X_delivery.shape[1]
neural_net_delivery = nnp.NeuralNetwork([0, m])
trained_thetas_delivery, error_hist_delivery = neural_net_delivery.train_logistic_regression(X_delivery, Y_delivery, 1, 10)