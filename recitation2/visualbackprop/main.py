# PyTorch tutorial codes for course Advanced Machine Learning
# main.py: trainig neural networks for MNIST classification
import time, datetime
from options import parser
from models import ConvNet

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torchvision import datasets, transforms

def visual_backprop(input, model, n_layers=3):
    ''' VisualBackprop '''
    masks = []
    x = input
    up_sample = torch.nn.Upsample(scale_factor=2)
    for idx, layer in enumerate(model.net):
        x = layer(x)
        _mask = x.mean(dim=1, keepdim = True)
        _max = torch.max(_mask, dim = 3, keepdim = True)[0]
        _max = torch.max(_max,  dim = 2, keepdim = True)[0]
        _min = torch.min(_mask, dim = 3, keepdim = True)[0]
        _min = torch.min(_min,  dim = 2, keepdim = True)[0]

        mask = (_mask - _min) / (_max - _min)
        masks += [mask]
        if idx >= n_layers:
            break
    cur_mask = masks[n_layers]
    for idx in range(n_layers-1, -1, -1):
        cur_mask = up_sample(cur_mask)* masks[idx]
    masked_input = cur_mask.repeat(1, 3, 1, 1) * input
    return masked_input

def get_dataloaders(args):
    ''' define training and testing data loader'''
    print('---Loading Data---')
    # load trainig data loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
            datasets.STL10('../Data/stl10', split='train', download=F, 
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
    model = ConvNet(use_batch_norm=True, use_resnet=False)
            
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
        loss = F.nll_loss(output, target)
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
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
           test_loss, correct, len(test_loader.dataset),
           100. * correct / len(test_loader.dataset)))

    masked_data = visual_backprop(data, model, n_layers=3)
    utils.save_image(data.data, 'origin_pictures.png', normalize=True, scale_each=True)
    utils.save_image(masked_data.data, 'visbackprop_pictures.png', normalize=True, scale_each=True)

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    print('\n--- Training ---')
    for epoch in range(1, args.epochs + 1):
        train(args, model, optimizer, train_loader, epoch)
        test(args, model, test_loader)
        scheduler.step()
        current_time = datetime.datetime.now().replace(microsecond=0)
        print('Time Interval:', current_time - start_time, '\n')
