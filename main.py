from activation_functions import ReLU, Sigmoid
from layers import Dense, Reshape, Convolutional, Pooling
from network import Network
import numpy as np

from pooling_functions import pooling_max


ai = Network([
    Reshape(784, (28, 28)),
    Convolutional((28, 28), ReLU, (5, 5), 3),
    Pooling((3, 24, 24), pooling_max, 2),
    Reshape((3, 12, 12), 432),
    Dense(432, 10, Sigmoid)
])

print(ai.run(np.random.rand(784)))