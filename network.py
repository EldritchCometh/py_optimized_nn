import numpy as np


class Activation:

    def __init__(self, act_func='tanh'):
        self.input = None
        self.output = None
        self.activation = None
        self.activation_prime = None
        self.set_act_funcs(act_func)

    def set_act_funcs(self, act_func):
        if act_func == 'tanh':
            self.activation = lambda x: np.tanh(x)
            self.activation_prime = lambda x: 1 - np.tanh(x) ** 2

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
        self.input = None
        self.output = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        if not np.array_equal(input, self.input):
            self.input = input
            self.output = np.expand_dims(input, -1)
            for layer in self.layers:
                self.output = layer.forward(self.output)
        return self.output


class Trainer:

    def __init__(self, network, training_samples):
        self.nn = network
        self.samples = training_samples

    def mse(self, y_true):
        return np.mean(np.power(y_true - self.nn.output, 2))

    def mse_prime(self, y_true):
        return 2 * (self.nn.output - y_true) / np.size(y_true)

    def backward(self, gradient, learning_rate):
        for layer in reversed(self.nn.layers):
            gradient = layer.backward(gradient, learning_rate)

    def train(self, learning_rate, epochs):
        for e in range(epochs):
            error = 0
            np.random.shuffle(self.samples)
            for (features, targets) in self.samples:
                self.nn.forward(features)
                error += self.mse(targets)
                output_gradient = self.mse_prime(targets)
                self.backward(output_gradient, learning_rate)
            error /= len(X)
            print('%d/%d, error=%f' % (e + 1, epochs, error))


if __name__ == "__main__":

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])
    samples = np.array(list(zip(X, Y)), dtype=object)

    nn = Network()
    nn.add_layer(Dense(2, 3))
    nn.add_layer(Activation('tanh'))
    nn.add_layer(Dense(3, 1))
    nn.add_layer(Activation('tanh'))

    Trainer(nn, samples).train(0.1, 1000)
