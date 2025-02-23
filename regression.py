""" Power Series Regression for single variable functions
"""

import sys
import numpy as np
from matplotlib import pyplot as plt

def normalize_data(data: np.ndarray) -> np.ndarray:
    """ Standardize inputs to have zero mean and unit variance. """
    return (data - np.mean(data)) / (np.std(data) + 1e-10)  # Avoid division by zero

# Denormalization function
def denormalize_data(norm_data, orig_data):
    return norm_data * np.std(orig_data) + np.mean(orig_data)


def forward_pass(coeff: list[float], inputs: np.ndarray[float]) -> np.ndarray[float]:
    """ Pass inputs into power series with given coefficients. Return outputs.
    """
    return sum([coeff[i] * np.clip(inputs ** i, 0, 1e6) for i in range(len(coeff))])

def compute_loss(model_outs: np.ndarray[float], acc_outs: np.ndarray[float]) -> float:
    """ Return the mean squared error of the model's predicted outputs
    """
    return np.mean(np.square(pred - acc) for pred, acc in zip(model_outs, acc_outs))

def compute_grad(inputs: np.ndarray[float], model_outs: np.ndarray[float], acc_outs: np.ndarray[float], deg: int) -> list[float]:
    """ Compute the gradient vector for the coefficients of the power series
    """
    n = len(inputs)
    grad = [(2 / n) * np.sum((model_outs - acc_outs) * np.clip(inputs**i, 0, 1e6)) for i in range(deg)]
    return np.clip(grad, -1e3, 1e3)

def regression(inputs: np.ndarray[float], acc_outs: np.ndarray[float], coeff: list[float], deg: int, iterations=10000, learning_rate=0.001) -> list[float]:
    """ Return a tuple of the optimized coefficients, and the r-squared value after the given number of
    training iterations. 
    """

    for _ in range(iterations):
        model_outs = forward_pass(coeff, inputs)
        grad = compute_grad(inputs, model_outs, acc_outs, deg)
        for i in range(deg):
            coeff[i] -= learning_rate * grad[i]

    y_mean = np.mean(acc_outs)
    ss_total = np.sum((acc_outs - y_mean) ** 2)
    ss_residual = np.sum((acc_outs - forward_pass(coeff, inputs)) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    return coeff + [r2]

if __name__ == "__main__":
    try:
        file_name = sys.argv[1]
        deg = int(sys.argv[2]) + 1
    except IndexError:
        print("Did not specify data file... program terminated.")
        sys.exit(1)

    xs, ys = np.array([]), np.array([])
    with open('clean_data/' + file_name + '.txt', "r") as f:
        lines = f.readlines()
        labels = lines[0]
        x_list, y_list = [], []
        for line in lines[1:]:
            data = line.rstrip().split(',')
            if data[0] != '' and data[1] != '':
                x_list.append(float(data[0]))
                y_list.append(float(data[1]))
        xs = np.array(x_list)
        ys = np.array(y_list)
    # Normalize your dataset before training

    orig_xs = xs.copy()
    orig_ys = ys.copy()

    xs = normalize_data(xs)
    ys = normalize_data(ys)

    # initialize coefficients
    coeff = [1] * deg

    # Regression
    data = regression(xs, ys, coeff, deg)
    coeff = data[:-1]
    r2 = data[-1]

    # Plot Data
    x_plot = np.linspace(np.min(orig_xs), np.max(orig_xs), 100)

    # Compute predicted y values using trained model
    y_norm_pred = forward_pass(coeff, (x_plot - np.mean(orig_xs)) / np.std(orig_xs))

    # Denormalize y values back to salary range
    y_plot = denormalize_data(y_norm_pred, orig_ys)

    plt.plot(x_plot, y_plot, color='r')
    plt.scatter(orig_xs, orig_ys, color='g')

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Years Worked")
    plt.ylabel("Salary")
    plt.title(f"Salary versus Years worked: R^2: {round(r2, 3)}")

    # To load the display window
    plt.show()
