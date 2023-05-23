import numpy as np


class Tanh:

    def __init__(self):
        self.input = None
        self.output = None
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
        print(self.weights)
        print(self.input)
        print(self.weights.dot(self.input))
        return self.weights.dot(self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = output_gradient.dot(self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)


X = np.array([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]])
Y = np.array([[[0]], [[1]], [[1]], [[0]]])
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Y = np.array([[0], [1], [1], [0]])

network = [
    Dense(2, 3),
    #Tanh(),
    #Dense(3, 1),
    #Tanh(),
]

output = X[1]
for layer in network:
    output = layer.forward(output)