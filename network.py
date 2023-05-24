import random
import numpy as np


class Tanh:

    def __init__(self):
        self.activation = lambda x: np.tanh(x)
        self.activation_prime = lambda x: 1 - np.tanh(x) ** 2

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Dense:

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)


class Network:

    def __init__(self, *structure):
        self.layers = structure

    def forward(self, input):
        input = np.expand_dims(input, -1)
        for layer in self.layers:
            input = layer.forward(input)
        return input


class Trainer:

    def __init__(self, neural_network, dataset):
        self.nn = neural_network
        self.dataset = dataset

    def mse(self, y_pred, y_true):
        return np.mean(np.power(y_true - y_pred, 2))

    def mse_prime(self, y_pred, y_true):
        return 2 * (y_pred - np.expand_dims(y_true, -1)) / np.size(y_true)

    def backward(self, gradient, learning_rate):
        for layer in reversed(self.nn.layers):
            gradient = layer.backward(gradient, learning_rate)

    def train(self, learning_rate, epochs):
        for e in range(epochs):
            error = 0
            random.shuffle(self.dataset)
            for x, y_true in self.dataset:
                y_pred = self.nn.forward(x)
                error += self.mse(y_pred, y_true)
                gradient = self.mse_prime(y_pred, y_true)
                self.backward(gradient, learning_rate)
            error /= len(self.dataset)
            print('%d/%d, error=%f' % (e + 1, epochs, error))


def xor_dataset():
    xor_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_Y = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])
    return list(zip(xor_X, xor_Y))


nn = Network(
    Dense(2, 3),
    Tanh(),
    Dense(3, 2),
    Tanh())
trainer = Trainer(nn, xor_dataset())
trainer.train(0.1, 5000)
