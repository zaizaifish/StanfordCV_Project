import numpy as np 
import torch
import torch.nn as nn
import time
import torchvision.models as models
import torch.nn.functional as F
def RESNET(x_train, y_train, x_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    # x_train = x_train.to(device)
    # x_test = x_test.to(device)
    # y_train = y_train.to(device)
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
            x = F.sigmoid(self.fc3(x))
            return x
        
    model = RESNET()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 100

    start_time = time.time()
    for t in range(num_epochs + 1):
        
        y_train_pred = model(x_train)
        
        loss = criterion(y_train_pred, y_train)
        if (t % 10 == 0): print("Epoch ", t, "MSE: ", loss.item())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print('y_train_pred.shape: ',y_train_pred.shape)
    training_time = time.time() - start_time
    print("Training time: {}".format(training_time))
    # make predictions
    y_test_pred = model(x_test)
    
    return y_test_pred