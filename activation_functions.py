import numpy as np

class Function:
    def fun(x):
        return x
    def derivative(x):
        return 1

class Sigmoid(Function):
    def fun(x):
        return 1/(1+np.exp(-x))
    def derivative(x):
        return np.exp(-x)/((1+np.exp(-x))**2)

class ReLU(Function):
    def fun(x):
        return np.maximum(x, 0.)
    def derivative(x):
        return np.where(x<0, 0., 1.)

