# PyTorch tutorial codes for course Advanced Machine Learning
# main.py: trainig neural networks for MNIST classification
import time, datetime
from options import parser
from models import Autoencoder

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torchvision import datasets, transforms

def get_dataloaders(args):
    ''' define training and testing data loader'''
    print('---Loading Data---')
    # load trainig data loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
            datasets.STL10('../Data/stl10', split='train', download=True, 
                            transform = transforms.Compose([
                            transforms.Resize(128),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])),
            batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    # load testing data loader
    test_loader = torch.utils.data.DataLoader(
            datasets.STL10('../Data/stl10', split='test', 
                            transform = transforms.Compose([
                            transforms.Resize(128),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])),
            batch_size=args.test_batch_size, shuffle=True, drop_last=True, **kwargs)
    
    return train_loader, test_loader

def get_model(args):
    ''' define model '''
    use_unet = True if args.model == 'UNet' else False
    model = Autoencoder(use_unet)
            
    print('---Model Information---')
    print('Net:', model)
    print('Use GPU:', args.use_cuda)
    return model.to(args.device)
	
def get_optimizer(args, model):
    ''' define optimizer '''
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('\n---Optimization Information---')
    print('optimizer: Adam')
    print('batch size:', args.batch_size)
    print('lr:', args.lr)
    return optimizer

def train(args, model, optimizer, train_loader, epoch):
    ''' define training function '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, test_loader):
    ''' define testing function '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += data.size(0)* F.mse_loss(output, data).item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    utils.save_image(data.data, 'origin_pictures.png', normalize=True, scale_each=False)
    utils.save_image(output.data,'reconstruct_pictures.png', normalize=True, scale_each=False)
        
if __name__ == '__main__':
    start_time = datetime.datetime.now().replace(microsecond=0)
    print('[Start at %s]'%start_time)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.use_cuda = use_cuda
    args.device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = get_dataloaders(args)
    model = get_model(args)
    optimizer = get_optimizer(args, model)
    
    print('\n--- Training ---')
    for epoch in range(1, args.epochs + 1):
        train(args, model, optimizer, train_loader, epoch)
        test(args, model, test_loader)
        current_time = datetime.datetime.now().replace(microsecond=0)
        print('Time Interval:', current_time - start_time, '\n')
