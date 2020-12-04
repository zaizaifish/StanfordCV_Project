import os
from PIL import Image 
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
def imagetoarray(path, label):
    trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    result = list()
    for name in os.listdir(path):
        filepath = path + '/' + name
        image = Image.open(filepath)
        image_arr = trans(image) 
        image_arr = image_arr.numpy()
        result.append((image_arr, label))
    return result

def imagetoarray2(path):
    trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
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
    train_clean = imagetoarray(path,0)

    path = '../dataset/train/messy'
    train_messy = imagetoarray(path,1)
    train = train_clean + train_messy

    path = '../dataset/val/clean'
    val_clean = imagetoarray(path,0)

    path = '../dataset/val/messy'
    val_messy = imagetoarray(path,1)
    val = val_clean + val_messy

    path = '../dataset/test'
    test = imagetoarray2(path)

    return [train, val, test]

def load(train, val, batch_size, repeat, shuffle):
        
    class TrainDataset(Dataset):
        def __init__(self, train, repeat):
            # TODO
            # 1. Initialize file path or list of file names.
            self.train = train
            self.repeat = repeat

        def __getitem__(self, index):
            # TODO
            # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
            # 2. Preprocess the data (e.g. torchvision.Transform).
            # 3. Return a data pair (e.g. image and label).
            image = self.train[index][0]
            label = np.array([self.train[index][1]])
            # label = self.train[index][1]
            # if (label == 0): label = np.array([0.9,0.1])
            # else: label = np.array([0.1,0.9])
            return image, label

        def __len__(self):
            # You should change 0 to the total size of your dataset.
            return len(self.train * self.repeat)

    
    train_data = TrainDataset(train = train, repeat = repeat)
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = shuffle)

    val_data = TrainDataset(train = val, repeat = repeat)
    val_loader = DataLoader(dataset = val_data, batch_size = batch_size, shuffle = shuffle)

    return [train_loader, val_loader]