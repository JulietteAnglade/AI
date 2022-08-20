import numpy as np
from math import sqrt

def pooling_max(x):
    return np.amax(x)

def pooling_min(x):
    return np.amin(x)

def pooling_average(x):
    return np.mean(x)

def pooling_L2(x):
    S=0
    for i in x:
        S+=i**2
    return sqrt(S)

