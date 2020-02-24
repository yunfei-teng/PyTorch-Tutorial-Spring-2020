# PyTorch tutorial codes for course Advanced Machine Learning
# models.py: define model structures
# read: https://pytorch.org/docs/stable/nn.html
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    ''' convolutional neural network '''
    def __init__(self):
        super(ConvNet, self).__init__()
        nf = 8
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, etc.)
        self.conv1 = nn.Conv2d(    1, nf* 1, 5, 1, 0) #24
        self.conv2 = nn.Conv2d(nf* 1, nf* 2, 4, 2, 1) #12
        self.conv3 = nn.Conv2d(nf* 2, nf* 4, 5, 1, 0) #8
        self.conv4 = nn.Conv2d(nf* 4, nf* 8, 4, 2, 1) #4
        self.conv5 = nn.Conv2d(nf* 8,    10, 4, 1, 0) #1
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x = torch.flatten(x, 1)
        return F.log_softmax(x, dim=1)

class FCNet(nn.Module):
    ''' fully-connected neural network '''
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100,  10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
