import numpy as np


class Layer:
    def __init__(self, dim_input, dim_output):
        self.dim_input=dim_input
        self.dim_output=dim_output
    
    def run(self, input):
        return np.zeros(self.dim_output)
    
    def learn(self, input_previous_layer, errors_next_layer):
        return np.zeros(self.dim_input)

class Dense(Layer):
    pass

class Convolutional(Layer):
    pass

class Pooling(Layer):
    pass

class Reshape(Layer):
    pass

