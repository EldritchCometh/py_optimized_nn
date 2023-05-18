import numpy as np


class Activation:

    def __init__(self):
        self.input = None
        self.output = None
        self.activation = None
        self.activation_prime = None

    def set_tanh(self):
        self.activation = lambda x: np.tanh(x)
        self.activation_prime = lambda x: 1 - np.tanh(x) ** 2

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Dense:

    def __init__(self, input_size, output_size):
        self.input = None
        self.output = None
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
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        output = None
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        return output


class Trainer:

    def __init__(self, network, training_samples):
        self.nn = network
        self.samples = training_samples

    @staticmethod
    def mse(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def mse_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)

    def backward(self, gradient, learning_rate):
        for layer in reversed(self.nn):
            gradient = layer.backward(gradient, learning_rate)

    def train(self, learning_rate, epochs):
        for e in range(epochs):
            error = 0
            samples = self.samples
            np.random.shuffle(samples)
            for sample in samples:
                features, targets = sample
                output = self.nn.forward(features)
                error += self.mse(targets, output)
                output_gradient = self.mse_prime(targets, output)
                self.backward(output_gradient, learning_rate)
            error /= len(X)
            print('%d/%d, error=%f' % (e + 1, epochs, error))


if __name__ == "__main__":

    X = np.array([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]])
    Y = np.array([[[0]], [[1]], [[1]], [[0]]])
    training_samples = np.concatenate((X, Y), axis=2)

    nn = Network()
    nn.add_layer(Dense(2, 3))
    nn.add_layer(Activation().set_tanh())
    nn.add_layer(Dense(3, 1))
    nn.add_layer(Activation().set_tanh())

    Trainer(nn, training_samples).train(0.1, 10000)
