#!/usr/bin/env python3
'''
Training a Perceptron Using Gradient Descent to Minimize Error
'''

from typing import List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

# load dataset
data = pd.read_csv("data.csv", header=None)

# Set random seed
np.random.seed(44)


# define X, y
X = np.array(data[[0, 1]])
y = np.array(data[2])


# Global Vars
weights_ = []
bias_ = []
errors_ = []
EPOCHS = 100
LEARN_RATE = 0.01
accuracy_ = []


# Activation (sigmoid) function
def sigmoid(input_x):
    """Sigmoid Function"""
    return 1 / (1 + np.exp(-input_x))


# Output (prediction) formula
def output_function(features, weights, bias):
    """Output Function"""
    return sigmoid(np.dot(features, weights) + bias)


# Error (log-loss) formula
def error_function(y_true, y_pred):
    """Error Function"""
    return -y * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)


# Gradient descent step
def update_weights(feature, y_true, weights, bias, learn_rate):
    """Update weights"""
    prediction = output_function(feature, weights, bias)
    d_error = y_true - prediction
    weights += learn_rate * d_error * feature
    bias += learn_rate * d_error
    return weights, bias


# Training percetron function
def train(features, targets, epochs, learn_rate, graph_lines=False):
    """Train Algorithm"""
    errors = []
    _, n_features = features.shape
    last_loss = None
    weights = np.random.normal(scale=1 / n_features**0.5, size=n_features)
    bias = 0
    for e_val in range(epochs):
        _ = np.zeros(weights.shape)
        for x_val, y_val in zip(features, targets):
            weights, bias = update_weights(
                x_val, y_val,
                weights, bias,
                learn_rate
            )

        # Printing out the log-loss error on the training set
        out = output_function(features, weights, bias)
        loss = np.mean(error_function(targets, out))
        errors.append(loss)
        if e_val % (epochs / 10) == 0:
            print("\n========== Epoch", e_val, "==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss

            # Converting the output (float) to boolean as
            # it is a binary classification
            # e.g. 0.95 --> True (= 1), 0.31 --> False (= 0)
            predictions = out > 0.5

            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
        if graph_lines and e_val % (epochs / 100) == 0:
            # display(-weights[0] / weights[1], -bias / weights[1])
            weights_.append(-weights[0] / weights[1])
            bias_.append(-bias / weights[1])
            errors_.append(loss)
            accuracy_.append(accuracy)

    weights_.append(-weights[0] / weights[1])
    bias_.append(-bias / weights[1])


# Call train function
train(X, y, EPOCHS, LEARN_RATE, True)


# Initialize Plots
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

ax[0].set_ylim(-0.05, 1.05)
ax[0].set_xlim(-0.05, 1.05)
ax[0].set_title("Solution Boundary")
ax[0].set_ylabel("Targets")
ax[0].set_xlabel("Features")

ax[1].set_xlim(0, EPOCHS)
ax[1].set_ylim(min(errors_), max(errors_))
ax[1].set_title("Error Plot (Gradient Descent)")
ax[1].set_ylabel("Error (Loss)")
ax[1].set_xlabel("Epoch")

admitted = X[np.argwhere(y == 1)]
rejected = X[np.argwhere(y == 0)]
ax[0].scatter(
    [s[0][0] for s in rejected],
    [s[0][1] for s in rejected],
    s=25,
    color="yellow",
    edgecolor="k",
    label="0",
)
ax[0].scatter(
    [s[0][0] for s in admitted],
    [s[0][1] for s in admitted],
    s=25,
    color="black",
    edgecolor="k",
    label="1",
)

text = ax[1].text(60, 0.6, "")
text2 = ax[1].text(0, errors_[0], f"Loss = {errors_[0]:.2f}")
err_ = np.zeros(len(errors_))
err_[:] = np.nan
(err_plt,) = ax[1].plot(err_, color="red", label="loss")

# Include legends to plots
ax[0].legend()
ax[1].legend()

# Animation vars
xs = np.arange(-10, 10, 0.1)
WEIGHT = 0
BIAS = 0


# Function to update plot
def update(i):
    '''Update frame funcion'''
    weight = weights_[i]
    bias = bias_[i]
    if i == len(weights_) - 1:
        line, = ax[0].plot(xs, weight * xs + bias, "black", lw=3)
    else:
        line, = ax[0].plot(xs, weight * xs + bias, "g--", alpha=0.1)
    try:
        acc = accuracy_[i]
        text.set_text(
            f"Weight = {weight:.4f},\nbia = {bias:.4f},\nAccuracy = {acc:.2f}"
        )
        text2.set_position((i, errors_[i]))
        text2.set_text(f"Loss = {errors_[i]:.2f}")
        err_[i] = errors_[i]
        err_plt.set_ydata(err_)
    except IndexError:
        pass

    return text, err_plt


# Maximize plot window
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")

# # Create the animation
ani = FuncAnimation(fig, update, frames=len(weights_), repeat=False)

# fig.title("Training A Perceptron Using Gradient Descent to Minimize Loss")
plt.show()
