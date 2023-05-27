import numpy as np
import gzip
import random
import pickle
import logging
import os
import time
import hashlib


class Activation:

    def __init__(self, activation_functions):
        self.activation, self.activation_prime = activation_functions

    @staticmethod
    def get_relu():
        activation = lambda x: np.maximum(0, x)
        activation_prime = lambda x: np.where(x > 0, 1, 0)
        return activation, activation_prime

    @staticmethod
    def get_tanh():
        activation = lambda x: np.tanh(x)
        activation_prime = lambda x: 1 - np.square(np.tanh(x))
        # activation_prime = lambda x: 1 - np.tanh(x) ** 2
        return activation, activation_prime

    @staticmethod
    def get_leaky_relu(alpha=0.01):
        activation = lambda x: np.where(x >= 0, x, alpha * x)
        activation_prime = lambda x: np.where(x >= 0, 1, alpha)
        return activation, activation_prime

    @staticmethod
    def get_identity():
        activation = lambda x: x
        activation_prime = lambda x: 1
        return activation, activation_prime

    @staticmethod
    def get_sigmoid():
        activation = lambda x: 1 / (1 + np.exp(-x))
        activation_prime = lambda x: activation(x) * (1 - activation(x))
        return activation, activation_prime

    @staticmethod
    def get_softmax():
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)
        def softmax_prime(x):
            s = softmax(x)
            return s * (1 - s)
        return softmax, softmax_prime

    def gradient_descent(self, learning_rate, batch_size):
        pass

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Dense:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * \
                       np.sqrt(2. / input_size)
        self.bias = np.random.randn(output_size, 1)
        self.weights_gradients = np.zeros((output_size, input_size))
        self.output_gradients = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient):
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights_gradients += weights_gradient
        self.output_gradients += output_gradient
        return np.dot(self.weights.T, output_gradient)

    def gradient_descent(self, learning_rate, batch_size):
        self.weights -= learning_rate * self.weights_gradients / batch_size
        self.bias -= learning_rate * self.output_gradients / batch_size
        self.weights_gradients = np.zeros_like(self.weights_gradients)
        self.output_gradients = np.zeros_like(self.output_gradients)


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
        self.training_data, self.testing_data = dataset

    # @staticmethod
    # def mse(y_pred, y_true):
        # return np.mean(np.power(y_true - y_pred, 2))

    # @staticmethod
    # def mse_prime(y_pred, y_true):
        # return 2 * (y_pred - np.expand_dims(y_true, -1)) / np.size(y_true)

    @staticmethod
    def cel(y_pred, y_true):
        y_true = np.expand_dims(y_true, -1)  # ensure correct shape
        return -np.sum(y_true * np.log(y_pred))

    @staticmethod
    def cel_prime(y_pred, y_true):
        y_true = np.expand_dims(y_true, -1)  # ensure correct shape
        return y_pred - y_true

    def backward(self, gradient):
        for layer in reversed(self.nn.layers):
            gradient = layer.backward(gradient)

    @staticmethod
    def classified_correctly(y_pred, y_true):
        return np.argmax(y_pred) == np.argmax(y_true)

    def evaluate_network(self, e, epochs):
        error = 0
        accuracy = 0
        for x, y_true in self.testing_data:
            y_pred = nn.forward(x)
            error += self.cel(y_pred, y_true)
            # error += self.mse(y_pred, y_true)
            accuracy += self.classified_correctly(y_pred, y_true)
        error /= len(self.testing_data)
        accuracy /= len(self.testing_data)
        print('%d/%d, error=%f, acc=%f' % (e + 1, epochs, error, accuracy))
        return accuracy

    def gradient_descent(self, learning_rate, batch_size):
        for layer in self.nn.layers:
            layer.gradient_descent(learning_rate, batch_size)

    def train(self, learning_rate, epochs, batch_size):
        highest_accuracy = 0
        highest_epoch = 0
        for e in range(epochs):
            random.shuffle(self.training_data)
            samples_length = len(self.training_data)
            mini_batches = [
                self.training_data[i:i + batch_size]
                for i in range(0, samples_length, batch_size)
                if i + batch_size <= samples_length]
            for mini_batch in mini_batches:
                for x, y_true in mini_batch:
                    y_pred = self.nn.forward(x)
                    gradient = self.cel_prime(y_pred, y_true)
                    # gradient = self.mse_prime(y_pred, y_true)
                    self.backward(gradient)
                self.gradient_descent(learning_rate, batch_size)
            accuracy = self.evaluate_network(e, epochs)
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                highest_epoch = e + 1
                print("new best")
        return highest_accuracy, highest_epoch


def xor_dataset():
    xor_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_Y = np.array([[0], [1], [1], [0]])
    return list(zip(xor_X, xor_Y)), list(zip(xor_X, xor_Y))


def mnist_dataset():
    with gzip.open('mnist_data.pkl.gz', 'rb') as f:
        return pickle.load(f)


script_dir = os.path.dirname(os.path.realpath(__file__))
log_file_path = os.path.join(script_dir, 'tests.log')
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(message)s')
while True:
    nn = Network(
        Dense(784, 16),
        Activation(Activation.get_tanh()),
        Dense(16, 16),
        Activation(Activation.get_tanh()),
        Dense(16, 10),
        Activation(Activation.get_softmax()))
    trainer = Trainer(nn, mnist_dataset())
    learning_rate = round(random.uniform(0.001, 0.1), 3)
    epochs = 100
    batch_size = random.choice([1, random.randint(2, 32)])
    accuracy, epoch = trainer.train(learning_rate, epochs, batch_size)
    formatted_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
    log_out = f'time:{formatted_time:<9} ' \
              f'lrate:{learning_rate:<6} ' \
              f'bsize:{batch_size:<3} ' \
              f'epoch:{epoch:<3} ' \
              f'acc:{accuracy:<6}'
    logging.info(log_out)
