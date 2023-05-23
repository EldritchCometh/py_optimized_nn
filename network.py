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

    def __init__(self):
        self.layers = [
            Dense(2, 3),
            Tanh(),
            Dense(3, 2),
            Tanh(),
        ]

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, gradient, learning_rate):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)


class Trainer:

    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def mse_prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)

    def train(self):
        epochs = 1000
        learning_rate = 0.1
        for e in range(epochs):
            error = 0
            for x, y in zip(X, Y):
                output = nn.forward(x)
                error += self.mse(y, output)
                gradient = self.mse_prime(y, output)
                nn.backward(gradient, learning_rate)
            error /= len(X)
            print('%d/%d, error=%f' % (e + 1, epochs, error))


X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0, 0], [1, 1], [1, 1], [0, 0]], (4, 2, 1))

nn = Network()
Trainer().train()
