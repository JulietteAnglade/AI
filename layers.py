from enum import Enum
import numpy as np


class Layer:
    def __init__(self, dim_input, dim_output):
        self.dim_input = dim_input
        self.dim_output = dim_output

    def run(self, input):
        return np.zeros(self.dim_output)

    def learn(self, input_previous_layer, errors_next_layer):
        return np.zeros(self.dim_input)


class Dense(Layer):
    def __init__(self, dim_input, dim_output, function, weights=None, biases=None, learning_rate=1):
        super().__init__(dim_input, dim_output)
        self.learning_rate = learning_rate
        self.function = function
        if weights is None:
            self.weights = np.random.rand(self.dim_output, self.dim_input) * 2 - 1
        else:
            self.weights = weights
        if biases is None:
            self.biases = np.random.rand(self.dim_output) * 2 - 1
        else:
            self.biases = biases

    def run(self, input):
        return self.function.fun(self.weights @ input + self.biases)

    def learn(self, input_previous_layer, weighted_errors_next_layer):
        errors = weighted_errors_next_layer * (
            self.function.derivative(self.weights @ input_previous_layer + self.biases))
        weighted_errors = np.transpose(self.weights) @ errors
        self.weights -= self.learning_rate * np.transpose(np.array([errors * node for node in input_previous_layer]))
        self.biases -= self.learning_rate * errors
        return weighted_errors


class Convolutional(Layer):
    def __init__(self, dim_input, function, filter_size, filter_number, stride=1, weights=None, biases=None,
                 learning_rate=1):
        dim_output = (filter_number, (dim_input[0] - (filter_size[0] - 1)), (dim_input[1] - (filter_size[1] - 1)))
        super().__init__(dim_input, dim_output)
        self.learning_rate = learning_rate
        self.filter_number = filter_number
        self.filter_size = filter_size
        self.function = function
        if weights is None:
            self.weights = np.array([np.random.rand(*filter_size) for _ in range(self.filter_number)])
        else:
            self.weights = weights
        if biases is None:
            self.biases = np.random.rand(self.filter_number)
        else:
            self.biases = biases

    def run(self, input):
        output = np.zeros(self.dim_output)
        for i, filter in enumerate(self.weights):
            for l in range(self.dim_input[0] - self.filter_size[0] + 1):
                for m in range(self.dim_input[1] - self.filter_size[1] + 1):
                    for k in range(self.filter_size[0]):
                        for j in range(self.filter_size[1]):
                            output[i][l][m] += filter[k][j] * input[l + k][m + j]
            output[i] += self.biases[i]
        return self.function.fun(output)


class Pooling(Layer):
    def __init__(self, dim_input, pooling_function, pooling_size):
        dim_output = (dim_input[0], dim_input[1] // pooling_size, dim_input[2] // pooling_size)
        super().__init__(dim_input, dim_output)
        self.pooling_function = pooling_function
        self.pooling_size = pooling_size

    def run(self, input):
        output = np.zeros(self.dim_output)
        for k in range(self.dim_input[0]):
            for l in range(0, self.dim_input[1] - self.pooling_size + 1, self.pooling_size):
                for m in range(0, self.dim_input[2] - self.pooling_size + 1, self.pooling_size):
                    list = []
                    for i in range(self.pooling_size):
                        for j in range(self.pooling_size):
                            list.append(input[k][l + i][m + j])
                    output[k][l // self.pooling_size][m // self.pooling_size] = self.pooling_function(list)
        return output


class Reshape(Layer):
    def __init__(self, dim_input, dim_output):
        super().__init__(dim_input, dim_output)

    def run(self, input):
        return np.reshape(input, self.dim_output)
