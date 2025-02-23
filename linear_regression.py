""" Linear Regression for single variable functions 
"""

import sys
import numpy as np
from matplotlib import pyplot as plt


def forward_pass(coeff: tuple[float, float], inputs: np.ndarray[float]) -> np.ndarray[float]:
    """ Pass inputs into linear function with given coefficients. Return outputs.
    """
    out = np.zeros(dtype=float, shape=inputs.shape)
    for i in range(inputs.shape[0]):
        out[i] = coeff[0] * inputs[i] + coeff[1]
    return out

def compute_loss(model_outs: np.ndarray[float], acc_outs: np.ndarray[float]) -> float:
    """ Return the mean squared error of the model's predicted outputs
    """
    return np.sqrt(sum(np.square(pred - acc) for pred, acc in zip(model_outs, acc_outs)))

def compute_grad(inputs: np.ndarray[float], model_outs: np.ndarray[float], acc_outs: np.ndarray[float]) -> tuple[float, float]:
    """ Compute the gradient vector for the coefficients of the linear function
    """
    loss = compute_loss(model_outs, acc_outs)
    a_grad = sum((pred - acc) * inp for pred, acc, inp in zip(model_outs, acc_outs, inputs)) / loss
    b_grad = sum(pred - acc for pred, acc in zip(model_outs, acc_outs)) / loss
    return a_grad, b_grad

def regression(inputs: np.ndarray[float], acc_outs: np.ndarray[float], coeff: tuple[float, float], iterations=100, learning_rate=0.5) -> tuple[float, float, float]:
    """ Return a tuple of the optimized coefficients, and the r-squared value after the given number of
    training iterations. 
    """

    for _ in iterations:
        model_outs = forward_pass(coeff, inputs)
        grad = compute_grad(inputs, model_outs, acc_outs)
        coeff[0] += -1 * grad[0] * learning_rate
        coeff[1] += -1 * grad[1] * learning_rate
    return coeff

if __name__ == "__main__":
    try:
        file_name = sys.argv[1]
    except IndexError:
        print("Did not specify data file... program terminated.")
        sys.exit(1)

    xs, ys = np.array([]), np.array([])
    with open(file_name, "r") as f:
        lines = f.readlines()
        x_list, y_list = [], []
        for line in lines:
            x_list.append(float(line[0]))
            y_list.append(float(line[1]))
        xs = np.array(x_list)
        ys = np.array(y_list)

    # initialize coefficients
    a = 1.0
    b = 0.0

    # Regression
    optimized_coeff = regression(xs, ys, (a, b))