import os
from PIL import Image 
import numpy as np
from dataloader import *
from svm import *
from resnet import *
train_clean, train_messy, val_clean, val_messy, test = parser()

x_train = np.concatenate((train_clean, train_messy), axis=0)
print("x_train.shape: ",x_train.shape)

# 1 for messy, 0 for clean
temp1 = np.zeros((train_clean.shape[0],1))
temp2 = np.ones((train_messy.shape[0],1))
y_train = np.concatenate((temp1, temp2), axis=0)
print("y_train.shape: ",y_train.shape)

x_val = np.concatenate((val_clean, val_messy), axis=0)
print("x_val.shape: ",x_val.shape)

temp1 = np.zeros((val_clean.shape[0],1))
temp2 = np.ones((val_messy.shape[0],1))
y_val = np.concatenate((temp1, temp2), axis=0)
print("y_val.shape: ",y_val.shape)

# y_train_pred, y_test_pred, training_time = SVM(x_train, y_train, test)
# print("Training time: {}".format(training_time))

# train_accuracy = (y_train.shape[0] - np.sum(np.abs(y_train.ravel() - y_train_pred))) / y_train.shape[0]

# print('Train Accuracy: ', train_accuracy)

# print('Test Result: ', y_test_pred) # should be {0,0,1,0,1,1,0,1,1,0}

# y_true = np.array([0,0,1,0,1,1,0,1,1,0])
# test_accuracy = (y_true.shape[0] - np.sum(np.abs(y_true - y_test_pred))) / y_true.shape[0]
# print('Test Accuracy: ', test_accuracy)

y_test_pred = RESNET(x_train, y_train, test)
print('y_test_pred: ',y_test_pred)