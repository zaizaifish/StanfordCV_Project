import os
from PIL import Image 
import numpy as np
import torchvision.transforms as transforms
import torch
def imagetoarray(path):
    trans=transforms.Compose([transforms.ToTensor()])
    result = list()
    for name in os.listdir(path):
        filepath = path + '/' + name
        image = Image.open(filepath)
        image_arr = trans(image) 
        image_arr = image_arr.numpy()
        result.append(image_arr)
    result = np.array(result)
    return result

def parser():
    # parse all data into numpy array
    path = '../dataset/train/clean'
    train_clean = imagetoarray(path)
    print("train_clean.shape: ",train_clean.shape)

    path = '../dataset/train/messy'
    train_messy = imagetoarray(path)
    print("train_messy.shape: ",train_messy.shape)

    path = '../dataset/val/clean'
    val_clean = imagetoarray(path)
    print("train_clean.shape: ",val_clean.shape)

    path = '../dataset/val/messy'
    val_messy = imagetoarray(path)
    print("train_messy.shape: ",val_messy.shape)

    path = '../dataset/test'
    test = imagetoarray(path)
    print("train_messy.shape: ",test.shape)

    return [train_clean, train_messy, val_clean, val_messy, test]
