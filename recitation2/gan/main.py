# PyTorch tutorial codes for course Advanced Machine Learning
# main.py: trainig neural networks for MNIST classification
import time, datetime
from options import parser
from models import  Generator, Discriminator

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torchvision import datasets, transforms

def get_dataloader(args):
    ''' define training and testing data loader'''
    # load trainig data loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./Data/mnist', train=True, download=True, 
                               transform = transforms.Compose([
                               transforms.Resize(32), 
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    return train_loader
	
def train(args, netG, netD, optimizerG, optimizerD, train_loader, epoch):
    ''' define training function '''
    # define input, noise, label
    input = torch.zeros(args.batch_size, 1, 32, 32, device = args.device)
    label = torch.zeros(args.batch_size, device = args.device)
    noise = torch.zeros(args.batch_size, args.nz, 1, 1 , device = args.device)
    real_label, fake_label = 1, 0
    criterion = torch.nn.BCELoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        # ---Discriminator
        netD.zero_grad()
        input.copy_(data)
        label.fill_(real_label)

        # real
        output = netD(input)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # fake
        noise.normal_()
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        # ---Generator
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        # ---Print
        if batch_idx % args.log_interval == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, args.epochs, batch_idx, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    # Save real and fake images
    utils.save_image(input.data,'real_samples.png', normalize=True, scale_each=False)
    utils.save_image(netG(noise).data,'fake_samples.png',normalize=True, scale_each=False)

        
if __name__ == '__main__':
    start_time = datetime.datetime.now().replace(microsecond=0)
    print('[Start at %s]'%start_time)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.use_cuda = use_cuda
    args.device = torch.device("cuda" if use_cuda else "cpu")

    train_loader = get_dataloader(args)
    netG = Generator(args.nz).to(args.device)
    netD = Discriminator().to(args.device)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)
    print(netG)
    print(netD)

    print('\n--- Training ---')
    for epoch in range(1, args.epochs + 1):
        train(args, netG, netD, optimizerG, optimizerD, train_loader, epoch)
        current_time = datetime.datetime.now().replace(microsecond=0)
        print('Time Interval:', current_time - start_time, '\n')
