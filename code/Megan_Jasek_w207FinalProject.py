# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:17:20 2016

@author: MeganJ
"""

#%matplotlib inline

import os
import time
import numpy as np
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
print(theano.config.device) # We're using CPUs (for now)
print(theano.config.floatX) # Should be 64 bit for CPUs

np.random.seed(0)

#### Constants
GPU = False
if GPU:
    print("Trying to run under a GPU.  If this is not desired, then set the GPU flag to False.")
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print ("Running with a CPU.  If this is not desired, then set the GPU flag to True.")

training_file_name = 'C:\\Megan\\Education\\Berkeley\\W207\\HW\\FinalProject-NN\\w207-Final-Project\\data\\training.csv'
test_file_name = 'C:\Megan\Education\Berkeley\W207\HW\FinalProject-NN\w207-Final-Project\data\test.csv'

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

num_train_examples = X1.shape[0]-500
train_data, train_labels = X1[:num_train_examples], Y1[:num_train_examples]
dev_data, dev_labels = X1[num_train_examples:], Y1[num_train_examples:]
print(train_data.shape)
print(dev_data.shape)

def RMSE (y, y_pred):
    return 48*sqrt(mean_squared_error(y, y_pred))

def RMSE_tensor (y, y_pred):
    return 48*T.pow(T.mean(T.pow(T.sub(y, y_pred),2)),0.5)

# try using the means
start_time = time.time()
mean_labels = np.asarray(np.mean(train_labels, axis=0))
predict =[]
for i in range(dev_labels.shape[0]):
    predict.append(mean_labels)
predict = np.asarray(predict)
print('Train time = %.2f' %(time.time() - start_time))
start_time = time.time()
RMSE_score = RMSE(dev_labels, predict)
print('RMSE = %.4f' %(RMSE_score))
print('Prediction time = %.2f' %(time.time() - start_time))
#print(lm.predict(dev_data))
#print()
#print(dev_labels)

# try Linear Regression
lm = LinearRegression()
start_time = time.time()
lm.fit(train_data, train_labels)
print('Train time = %.2f' %(time.time() - start_time))
start_time = time.time()
accuracy = lm.score(dev_data, dev_labels)
RMSE_score = RMSE(dev_labels, lm.predict(dev_data))
print('Accuracy = %.4f' %(accuracy))
print('RMSE = %.4f' %(RMSE_score))
print('Prediction time = %.2f' %(time.time() - start_time))
#print(lm.predict(dev_data))
#print()
#print(dev_labels)

# Make a 1-layer fully-connected layer
numFeatures = train_data.shape[1]
numClasses = train_labels.shape[1]

## (1) Parameters 
# Initialize the weights to small, but non-zero, values.
w = theano.shared(np.asarray((np.random.randn(*(numFeatures, numClasses))*.01)))

## (2) Model
# Theano objects accessed with standard Python variables
X = T.matrix()
Y = T.matrix()

def model(X, w):
    #return T.nnet.softmax(T.dot(X, w))
    return T.dot(X, w)
y_hat = model(X, w)

## (3) Cost function
#cost = T.mean(T.nnet.categorical_crossentropy(y_hat, Y))
cost = RMSE_tensor(Y, y_hat)

## (4) Objective (and solver)

alpha = 0.01
gradient = T.grad(cost=cost, wrt=w) 
update = [[w, w - gradient * alpha]] 
train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True) # computes cost, then runs update
#y_pred = T.argmax(y_hat, axis=1) # select largest probability as prediction
y_pred = y_hat
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

def gradientDescent(epochs):
    trainTime = 0.0
    predictTime = 0.0
    for i in range(epochs):
        start_time = time.time()
        cost = train(train_data[0:len(train_data)], train_labels[0:len(train_data)])
        trainTime =  trainTime + (time.time() - start_time)
        #print('%d) accuracy = %.4f' %(i+1, np.mean(dev_labels == predict(dev_data))))
        print('%d) RMSE = %.4f' %(i+1, RMSE(dev_labels, predict(dev_data))))
    print('train time = %.2f' %(trainTime))

gradientDescent(50)

start_time = time.time()
print(predict(dev_data))
print()
print(dev_labels)
print('predict time = %.2f' %(time.time() - start_time))

miniBatchSize = 10 
def gradientDescentStochastic(epochs):
    trainTime = 0.0
    predictTime = 0.0
    start_time = time.time()
    for i in range(epochs):       
        for start, end in zip(range(0, len(train_data), miniBatchSize), range(miniBatchSize, len(train_data), miniBatchSize)):
            cost = train(train_data[start:end], train_labels[start:end])
        trainTime =  trainTime + (time.time() - start_time)
        #print('%d) accuracy = %.4f' %(i+1, np.mean(np.argmax(test_labels_b, axis=1) == predict(test_data))))
        print('%d) RMSE = %.4f' %(i+1, RMSE(dev_labels, predict(dev_data))))
    print('train time = %.2f' %(trainTime))
    
gradientDescentStochastic(50)

start_time = time.time()
print(predict(dev_data))
print()
print(dev_labels)   
print('predict time = %.2f' %(time.time() - start_time))

numFeatures = train_data.shape[1]
numClasses = train_labels.shape[1]

## (1) Parameters
numHiddenNodes = 600 
w_1 = theano.shared(np.asarray((np.random.randn(*(numFeatures, numHiddenNodes))*.01)))
w_2 = theano.shared(np.asarray((np.random.randn(*(numHiddenNodes, numClasses))*.01)))
params = [w_1, w_2]


## (2) Model
X = T.matrix()
Y = T.matrix()
# Two notes:
# First, feed forward is the composition of layers (dot product + activation function)
# Second, activation on the hidden layer still uses sigmoid
def model(X, w_1, w_2):
    #return T.nnet.softmax(T.dot(T.nnet.sigmoid(T.dot(X, w_1)), w_2))
    return T.dot(T.nnet.sigmoid(T.dot(X, w_1)), w_2)
y_hat = model(X, w_1, w_2)

## (3) Cost...same as logistic regression
#cost = T.mean(T.nnet.categorical_crossentropy(y_hat, Y))
cost = RMSE_tensor(Y, y_hat)

## (4) Minimization.  Update rule changes to backpropagation.
alpha = 0.01
def backprop(cost, w):
    grads = T.grad(cost=cost, wrt=w)
    updates = []
    for w1, grad in zip(w, grads):
        updates.append([w1, w1 - grad * alpha])
    return updates
update = backprop(cost, params)
train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
#y_pred = T.argmax(y_hat, axis=1)
y_pred = y_hat
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

miniBatchSize = 1
def gradientDescentStochastic(epochs):
    trainTime = 0.0
    predictTime = 0.0
    start_time = time.time()
    for i in range(epochs):
        for start, end in zip(range(0, len(train_data), miniBatchSize), range(miniBatchSize, len(train_data), miniBatchSize)):
            cost = train(train_data[start:end], train_labels[start:end])
        trainTime =  trainTime + (time.time() - start_time)
        #print('%d) accuracy = %.4f' %(i+1, np.mean(np.argmax(test_labels_b, axis=1) == predict(test_data))))
        print('%d) RMSE = %.4f' %(i+1, RMSE(dev_labels, predict(dev_data))))
    print('train time = %.2f' %(trainTime))

gradientDescentStochastic(5)

start_time = time.time()
print(predict(dev_data))
print()
print(dev_labels)  
print('predict time = %.2f' %(time.time() - start_time))