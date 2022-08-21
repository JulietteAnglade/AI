import data
import loss_functions
from activation_functions import ReLU, Sigmoid
from layers import Dense, Reshape, Convolutional, Pooling
from network import Network
import numpy as np

from pooling_functions import pooling_max


def test(dataset, labels, ai):
    correctanswers=0
    for i,inputs in enumerate(dataset):
        output=ai.run(inputs)
        imax=0
        for j in range(len(output)):
            if output[j]>output[imax]:
                imax=j
        if imax == labels[i]:
            correctanswers+=1
    print(f"correct answer rate ={(correctanswers/len(dataset))*100} %")


ai = Network([
    Dense(784, 32, Sigmoid),
    Dense(32, 16, Sigmoid),
    Dense(16, 10, Sigmoid),
], loss_functions.quadratic_derivative)

images_training = data.load_images("dataset/train-images.idx3-ubyte")
target_training = data.load_targets("dataset/train-labels.idx1-ubyte")
images_test = data.load_images("dataset/t10k-images.idx3-ubyte")
label_test = data.load_labels("dataset/t10k-labels.idx1-ubyte")

for i in range(1000):
    print(i)
    ai.train(images_training, target_training, 100)
    data.save_ai(ai, "test.ai")
    if i % 100 == 0:
        test(images_test, label_test, ai)

test(images_test, label_test, ai)

