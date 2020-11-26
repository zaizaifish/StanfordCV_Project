import numpy as np 
import torch
import torch.nn as nn
import time
import torchvision.models as models
import torch.nn.functional as F

def RESNET(train_loader, val_loader, test, num_epochs):
    class RESNET(nn.Module):
        def __init__(self):
            super(RESNET, self).__init__()
            
            self.resnet = models.resnet18(pretrained=False)

            self.fc1 = nn.Linear(1000, 32)
            self.fc2 = nn.Linear(in_features=32, out_features=16)
            self.fc3 = nn.Linear(in_features=16, out_features=2)

        def forward(self, x):
            x = self.resnet(x)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.softmax(self.fc3(x), dim = -1)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)
    model = RESNET().to(device)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    start_time = time.time()

    for t in range(num_epochs + 1):
        loss_record = list()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)                                
            loss = criterion(outputs.float(), labels.float())
            loss_record.append(loss.item()) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print("Epoch ", t, "MSE: ", max(loss_record))

    training_time = time.time() - start_time
    print("Training time: {}".format(training_time))

    # make predictions
    test = torch.from_numpy(test).type(torch.Tensor)
    test = test.to(device)
    y_test_pred = model(test)
    print('Test Prob: {}'.format(y_test_pred))
    _, res = torch.max(y_test_pred,1)
    res = res.cpu().detach()
    res = res.numpy()
    return res