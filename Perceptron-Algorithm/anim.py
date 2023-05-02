import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
np.random.seed(100)


def step_function(t):
    return 1 if t >= 0 else 0


def prediction(X, W, b):
    return step_function((np.matmul(X, W) + b)[0])


def perceptronStep(X, y, W, b, learn_rate=0.01):
    for i in range(len(X)):
        yhat = prediction(X[i], W, b)
        delta = y[i] - yhat
        if delta == 1:
            W[0] += learn_rate * X[i][0]
            W[1] += learn_rate * X[i][1]
            b += learn_rate
        elif delta == -1:
            W[0] -= learn_rate * X[i][0]
            W[1] -= learn_rate * X[i][1]
            b -= learn_rate
    return W, b


def trainPerceptronAlgorithm(X, y, learn_rate=0.01, num_epochs=25):
    '''This function runs the perceptron algorithm repeatedly on the dataset,
    and returns a few of the boundary lines obtained in the iterations,
    for plotting purposes.'''

    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines


data = np.loadtxt("data.csv", delimiter=",")


boundary_lines = trainPerceptronAlgorithm(
    data[:, [0, 1]],
    data[:, 2],
    learn_rate=0.1,
    num_epochs=70
)


fig, ax = plt.subplots()
scatter = ax.scatter(
    data[:, 0],
    data[:, 1],
    c=data[:, 2],
    marker="o",
    edgecolors="black"
)
scatter.set_zorder(10)
pl, = ax.plot([], [], "black")
text = ax.text(0.1, 0.9, "")


ax.set_xlabel('x1')
ax.set_ylabel('x2')


# Function to update plot
def update(i):
    line = boundary_lines[i]
    x_vals = data[:, [0, 1]]
    y_vals = line[0] * x_vals + line[1]
    pl.set_xdata(x_vals)
    pl.set_ydata(y_vals)
    text.set_text(f"Weight = {line[0]}, bia = {line[1]}")
    return pl, text


# Maximize plot window
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')

# Create the animation
ani = FuncAnimation(fig, update, frames=len(boundary_lines), repeat=False)

# Show the plot
plt.title("Training a Perceptron to Find Best Line of Fit")
plt.show()
