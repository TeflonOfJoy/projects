


# manage imports
from random import randrange
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Softmax
from keras.regularizers import l1, l2
import numpy as np



# generate data
ZERO = np.array(
[[[0], [0], [0], [0], [0], [0], [0], [0]],
 [[0], [0], [0], [1], [1], [0], [0], [0]],
 [[0], [0], [1], [0], [0], [1], [0], [0]],
 [[0], [0], [1], [0], [0], [1], [0], [0]],
 [[0], [0], [1], [0], [0], [1], [0], [0]],
 [[0], [0], [1], [0], [0], [1], [0], [0]],
 [[0], [0], [0], [1], [1], [0], [0], [0]],
 [[0], [0], [0], [0], [0], [0], [0], [0]]])

ONE = np.array(
[[[0], [0], [0], [0], [0], [0], [0], [0]],
 [[0], [0], [0], [1], [1], [0], [0], [0]],
 [[0], [0], [1], [1], [1], [0], [0], [0]],
 [[0], [0], [0], [1], [1], [0], [0], [0]],
 [[0], [0], [0], [1], [1], [0], [0], [0]],
 [[0], [0], [0], [1], [1], [0], [0], [0]],
 [[0], [0], [1], [1], [1], [1], [0], [0]],
 [[0], [0], [0], [0], [0], [0], [0], [0]]])

DIM = ZERO.shape
ZEROES = 20
ONES = 20
NOISE = 0.05

def generate_data(zeroes = ZEROES, ones = ONES, noise = NOISE):
    x = []
    y = []
    while zeroes + ones > 0:
        if randrange(zeroes + ones) < zeroes:
            zeroes -= 1
            x.append(np.mod(ZERO + np.random.choice([0, 1], size = DIM, p = [1 - NOISE, NOISE]), 2))
            y.append(0)
        else:
            ones -= 1
            x.append(np.mod(ONE + np.random.choice([0, 1], size = DIM, p = [1 - NOISE, NOISE]), 2))
            y.append(1)
    return np.array(x), np.array(y)



# train neural network
FILTERS = 12
KERNEL = (3, 3)
REGULARIZER = l1(0.01)

EPOCHS = 10
VALIDATION = 0.1

# this method should train and return a neural network with a convolutional and maxpooling layer on the data x, y;
# use the above constants as parameters in the network definition; don't forget to add regularization
def train_network(x, y):
    # TODO: implement
    pass



# main script
x, y = generate_data()
network = train_network(x, y)


