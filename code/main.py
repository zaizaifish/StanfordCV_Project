import os
from PIL import Image 
import numpy as np
from dataloader import *
from svm import *
from resnet import *
from cnn import *
# 1 for messy, 0 for clean
# should be {0,0,1,0,1,1,0,1,1,0}
train, val, test = parser()
# result_val, result_test = SVM(train, val, test)
# parameters
batch_size = 10
repeat = 1
shuffle = True
num_epochs = 10
learning_rate = 0.001
train_loader, val_loader = load(train, val, batch_size, repeat, shuffle)
# result = RESNET(train_loader, val_loader, test, num_epochs, learning_rate)
result = CNN(train_loader, val_loader, test, num_epochs, learning_rate)
print('Test Result: {}'.format(result))
true = np.array([0,0,1,0,1,1,0,1,1,0])
accuracy = 1 - np.sum(np.abs(true - result)) / result.shape[0]
print('Accuracy Result: {}'.format(round(accuracy,2)))
# print('Val Result: {}'.format(round(result_val,2)))
# print('Test Result: {}'.format(round(result_test,2)))
