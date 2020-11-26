import os
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image 
import time
def imagetoarray(path, label):
    trans=transforms.Compose([transforms.ToTensor()])
    result = list()
    for name in os.listdir(path):
        filepath = path + '/' + name
        image = Image.open(filepath)
        image_arr = trans(image) 
        image_arr = image_arr.numpy()
        result.append((image_arr, label))
    return result

path = '../dataset/train/clean'
train_clean = imagetoarray(path,0)


path = '../dataset/train/messy'
train_messy = imagetoarray(path,1)

train = train_clean + train_messy
print(len(train))

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
        return image, label

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.train * self.repeat)

batch_size = 10
repeat = 1
shuffle = True
train_data = TrainDataset(train = train, repeat = repeat)
train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = shuffle)

# for i, data in enumerate(train_loader, 0):
#     # get the inputs; data is a list of [inputs, labels]
#     inputs, labels = data
#     print(inputs.shape)
#     print(labels.shape)

class RESNET(nn.Module):
    def __init__(self):
        super(RESNET, self).__init__()
        
        self.resnet = models.resnet18(pretrained=True)

        self.fc1 = nn.Linear(1000, 32)
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)
model = RESNET().to(device)
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 1
start_time = time.time()

for t in range(num_epochs + 1):
    loss_record = 0 
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)                                
        loss = criterion(outputs.float(), labels.float())
        loss_record = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (t % 10 == 0): print("Epoch ", t, "MSE: ", loss_record)

training_time = time.time() - start_time
print("Training time: {}".format(training_time))


