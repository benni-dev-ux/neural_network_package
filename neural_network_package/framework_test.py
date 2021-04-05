# import our machine learning framework
import neural_network_package as nnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import easygui
import pickle


# Load via file picker:
TESTDATA_FILE_PATH = 'C:/Users/Lora/Documents/ML/neural_net_gesture_detection/training_data/csvs/training_sets/own_combined_all_old.csv'#easygui.fileopenbox()


frames = pd.read_csv(TESTDATA_FILE_PATH)
print(TESTDATA_FILE_PATH)

# Define Features used for Gesture Detection
X = frames[
    ["RShoulder_x", "RShoulder_y", "LShoulder_x", "LShoulder_y", "RElbow_x", "RElbow_y", "LElbow_x", "LElbow_y"]].values
frames[frames["ground_truth"] == "flip"] = 9
frames[frames["ground_truth"] == "spread"] = 8
frames[frames["ground_truth"] == "pinch"] = 7
frames[frames["ground_truth"] == "swipe_down"] = 6
frames[frames["ground_truth"] == "swipe_up"] = 5
frames[frames["ground_truth"] == "rotate_right"] = 4
frames[frames["ground_truth"] == "rotate_left"] = 3
frames[frames["ground_truth"] == "rotate"] = 3
frames[frames["ground_truth"] == "swipe_right"] = 2
frames[frames["ground_truth"] == "swipe_left"] = 1
frames[frames["ground_truth"] == "idle"] = 0

# Scale X and add Bias
scaler = nnp.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_scaled = nnp.add_bias_column(X_scaled)

# Define Y as identity Matrix from the ground truth
Y = np.identity(10, dtype=int)[
    frames[["ground_truth"]].astype(int).values.flatten()]

# initialize a new neural net with a specific shape
neural_net = nnp.NeuralNetwork([8, 20, 20, 10])

# Define Optimisation Parameters
num_samples = len(X)
alpha = 0.3
iterations = 100
batch_size = 16
lambda_value = 0.3
beta_value = 0.95  # should be between 0.5 and 0.99

# train the Neural Net
error_history, accuracy_history, trained_thetas, softmax = neural_net.train(
    X_scaled[:num_samples], Y[:num_samples], alpha, batch_size, iterations, lambda_value, beta_value)

#%% print
#plotting accuracy and error history for test
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(np.arange(iterations), accuracy_history)
ax1.set(title="Accuracy history")

ax2.plot(np.arange(iterations), error_history)
ax2.set(title="Error history")

plt.show()
