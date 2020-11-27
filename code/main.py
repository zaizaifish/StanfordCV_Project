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
# y_test_pred = SVM(train, val, test)
# parameters
batch_size = 10
repeat = 1
shuffle = True
num_epochs = 10
learning_rate = 0.001
train_loader, val_loader = load(train, val, batch_size, repeat, shuffle)
result = RESNET(train_loader, val_loader, test, num_epochs, learning_rate)
# result = CNN(train_loader, val_loader, test, num_epochs)
print('Test Result: {}'.format(result))