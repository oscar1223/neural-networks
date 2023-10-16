# Neural network from scratch

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('csv/train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

# Inicializa los parametros
def init_params():
    W1 = np.random.randn(10, 784)
    b1 = np.random.randn(10, 1)
    W2 = np.random.randn(10, 10)
    b2 = np.random.randn(10, 1)

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def fordward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(A1)

    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_y = np.zeros((Y.size, Y.max()+1))
    one_hot_y[np.arange(Y.size), Y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

def deriv_ReLU():
    pass

def back_prop(Z1, A1, Z2, A2, W2, Y):
     m = Y.size
     one_hot_y = one_hot(Y)
     dZ2 = A2 - one_hot_y
     dW2 = 1 / m * dZ2.dot(A1.T)
     db2 = 1 / m * np.sum(dZ2, 2)
     dZ1 = W2.T.dot(dZ2)
