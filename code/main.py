import os
from PIL import Image 
import numpy as np
from dataloader import *

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
