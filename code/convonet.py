# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 07:20:14 2016

@author: MeganJ
"""

#%matplotlib inline

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import theano 
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

print(theano.config.device) # We're using CPUs (for now)
print(theano.config.floatX) # Should be 64 bit for CPUs

np.random.seed(0)

print(theano.config.device) # We're using CPUs (for now)
print(theano.config.floatX) # Should be 64 bit for CPUs

np.random.seed(0)

#### Constants
GPU = True
if GPU:
    print("Trying to run under a GPU.  If this is not desired, then set the GPU flag to False.")
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print ("Running with a CPU.  If this is not desired, then set the GPU flag to True.")

#training_file_name = 'C:\\Megan\\Education\\Berkeley\\W207\\HW\\FinalProject-NN\\training.csv'
#test_file_name = 'C:\Megan\Education\Berkeley\W207\HW\FinalProject-NN\w207-Final-Project\data\test.csv'

training_file_name = '//data//w207-Final-Project//data//training.csv'
test_file_name = '//data//w207-Final-Project//data//test.csv'

#training data:  7049 rows x 31 columns
# load pandas dataframe
df_train = read_csv(training_file_name)
# drop all rows that have missing values in them
# 2284 images left in training data
df_train = df_train.dropna()
#print(df_train['Image'])
# take the pandas Series and convert each row to a np.array
df_train['Image'] = df_train['Image'].apply(lambda im: np.fromstring(im, dtype=np.float32, sep=' '))
#df_train = df_train.dropna()

#print(df_train.count())  # prints the number of values for each column

# Convert pandas Series in to np.array for size (2140, 9216)
X1 = np.vstack(df_train['Image'].values) / 255.  # scale pixel values to [0, 1]
# this was done up top when converting to the strings to np.array
print(X1.shape)

# RESHAPE TO 2D
# reshapes X to a 2140 row array of 96x96 arrays, but the 1 adds and extra array in there, I'm
# not sure why
#X = X.reshape(-1, 1, 96, 96)
#print(X)
#print(X.shape)

Y1 = df_train[df_train.columns[:-1]].values
# scale target coordinates to [-1, 1]
Y1 = (Y1 - 48) / 48 
Y1 = Y1.astype(np.float32)
print(Y1.shape)

# shuffle train data
X1, Y1 = shuffle(X1, Y1, random_state=14)

#shared_X = theano.shared(np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
#shared_y = theano.shared(np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
#shared_X = theano.shared(X.astype(theano.config.floatX), borrow=True)
#shared_y = theano.shared(y.astype(theano.config.floatX), borrow=True)
# the types should be set properly already, so don't need to do the above
shared_X = theano.shared(X1, borrow=True)
shared_Y = theano.shared(Y1, borrow=True)
#print(shared_X)
#print(shared_Y)
#print(shared_Y.get_value)

num_dev_examples = 340
num_train_examples = X1.shape[0]-num_dev_examples
train_data, train_labels = X1[:num_train_examples], Y1[:num_train_examples]
dev_data, dev_labels = X1[num_train_examples:], Y1[num_train_examples:]
print(train_data.shape)
print(dev_data.shape)

def RMSE (y, y_pred):
    return 48*sqrt(mean_squared_error(y, y_pred))

def RMSE_tensor (y, y_pred):
    return 48*T.pow(T.mean(T.pow(T.sub(y, y_pred),2)),0.5)

def MSE (y, y_pred):
    return mean_squared_error(y, y_pred)

def MSE_tensor (y, y_pred):
    return T.mean(T.pow(T.sub(y, y_pred),2))

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

def plot_samples(num_samples, X_images, Y_pred):
    #X, _ = load(test=True)
    #y_pred = net1.predict(X)
    #X_image = dev_data
    #Y_pred = predict(dev_data)

    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(num_samples):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X_images[i], Y_pred[i], ax)

    plt.show()
    

# Adding convolutional layers

theano.config.floatX = 'float64'

numFeatures = train_data.shape[1]
numClasses = train_labels.shape[1]

## (1) Parameters
numHiddenNodes = 500 
patchWidth = 3
patchHeight = 3
featureMapsLayer1 = 32
featureMapsLayer2 = 64
featureMapsLayer3 = 128

# For convonets, we will work in 2d rather than 1d.  The images are 96x96 in 2d.
imageWidth = 96
train_data_2d = train_data.reshape(-1, 1, imageWidth, imageWidth)
#test_data_2d = test_data.reshape(-1, 1, imageWidth, imageWidth)
dev_data_2d = dev_data.reshape(-1, 1, imageWidth, imageWidth)

# Convolution layers.  
w_1 = theano.shared(np.asarray((np.random.randn(*(featureMapsLayer1, 1, patchWidth, patchHeight))*.01)))
w_2 = theano.shared(np.asarray((np.random.randn(*(featureMapsLayer2, featureMapsLayer1, patchWidth, patchHeight))*.01)))
w_3 = theano.shared(np.asarray((np.random.randn(*(featureMapsLayer3, featureMapsLayer2, patchWidth, patchHeight))*.01)))

# Fully connected NN. 
w_4 = theano.shared(np.asarray((np.random.randn(*(featureMapsLayer3 * 10 * 10, numHiddenNodes))*.01)))
w_5 = theano.shared(np.asarray((np.random.randn(*(numHiddenNodes, numClasses))*.01)))
params = [w_1, w_2, w_3, w_4, w_5]

## (2) Model
#X = T.matrix()
X = T.tensor4(dtype=theano.config.floatX) # conv2d works with tensor4 type
Y = T.matrix()

srng = RandomStreams()
def dropout(X, p=0.):
    if p > 0:
        X *= srng.binomial(X.shape, p=1 - p)
        X /= 1 - p
    return X

# Two notes:
# First, feed forward is the composition of layers (dot product + activation function)
# Second, activation on the hidden layer still uses sigmoid

# Theano provides built-in support for add convolutional layers
def model(X, w_1, w_2, w_3, w_4, w_5, p_1, p_2):
    l1 = dropout(max_pool_2d(T.maximum(conv2d(X, w_1, border_mode='full'),0.), (2, 2), ignore_border=True), p_1)
    l2 = dropout(max_pool_2d(T.maximum(conv2d(l1, w_2), 0.), (2, 2), ignore_border=True), p_1)
    l3 = dropout(T.flatten(max_pool_2d(T.maximum(conv2d(l2, w_3), 0.), (2, 2), ignore_border=True), outdim=2), p_1) # flatten to switch back to 1d layers
    l4 = dropout(T.maximum(T.dot(l3, w_4), 0.), p_2)
    return T.dot(l4, w_5)

#y_hat = model(X, w_1, w_2)
y_hat_train = model(X, w_1, w_2, w_3, w_4, w_5, 0.2, 0.5)
y_hat_predict = model(X, w_1, w_2, w_3, w_4, w_5, 0., 0.)
#y_x = y_hat

## (3) Cost
#cost = T.mean(T.nnet.categorical_crossentropy(y_hat, Y))
#cost = T.mean(T.nnet.categorical_crossentropy(y_hat_train, Y))
cost = MSE_tensor(Y, y_hat_train)

## (4) Minimization.  Update rule changes to backpropagation.
def backprop(cost, w, alpha=0.01, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=w)
    updates = []
    for w1, grad in zip(w, grads):
        
        # adding gradient scaling
        acc = theano.shared(w1.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * grad ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        grad = grad / gradient_scaling
        updates.append((acc, acc_new))
        
        updates.append((w1, w1 - grad * alpha))
    return updates

update = backprop(cost, params)
train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
#y_pred = T.argmax(y_hat_predict, axis=1)
y_pred = y_hat_predict
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

miniBatchSize = 1
def gradientDescentStochastic(epochs):
    trainTime = 0.0
    predictTime = 0.0
    start_time = time.time()
    for i in range(epochs):
        for start, end in zip(range(0, len(train_data_2d), miniBatchSize), range(miniBatchSize, len(train_data_2d), miniBatchSize)):
            cost = train(train_data_2d[start:end], train_labels[start:end])
        trainTime =  trainTime + (time.time() - start_time)
        D_RMSE = RMSE(dev_labels, predict(dev_data_2d))
        T_MSE = MSE(train_labels, predict(train_data_2d))
        D_MSE = MSE(dev_labels, predict(dev_data_2d))
        print('%d) RMSE = %.4f, T_MSE = %.4f, D_MSE = %.4f, T_MSE/D_MSE = %.4f' % (i+1, D_RMSE, T_MSE, D_MSE, T_MSE/D_MSE))
    print('train time = %.2f' %(trainTime))

gradientDescentStochastic(400)

start_time = time.time()
#plot_samples(16, dev_data_2d, predict(dev_data_2d))
print(predict(dev_data_2d))
print()
print(dev_labels)  
print('predict time = %.2f' %(time.time() - start_time))