#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:19:50 2019

@author: karrington
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:19:05 2019

@author: karrington
"""

import matplotlib.pyplot as plt
import argparse
import itertools
import torch
import numpy as np
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import displayVAE
from displayVAE import display_images

import uuid

#from displayVAE import acquire_data_hook


import sys

dldb_path = '/home/karrington/git.workspace/DLDB2.0'  # default for Karrington, in her home area
                                                        # on muddlehead

if uuid.getnode() == 25965843724714:  # Bill's machine, different path
    dldb_path = '/home/bill/Desktop/Desktop/work/Brent Lab/Boucheron CNNs/DLDBproject'

    
if dldb_path not in sys.path[1]:
    sys.path.insert(1,dldb_path)


import dldb
from dldb import dlTile
from dldb import BioImage
import time
import openslide
import lmdb
#import dldb





parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "aya")



kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    BioImage(),batch_size=128)
#train_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../data', train=True, download=True,
#                   transform=transforms.ToTensor()),
#    
#test_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#    batch_size=args.batch_size, shuffle=True, **kwargs))


#def grad_hook(m,get,give):
#    global GradInput
#    GradInput = get
#    global GradOutput
#    GradOutput= give
#      
T2=2500

class VAE(nn.Module):
    def __init__(self,dim=20):  # this sets up 5 linear layers
        super(VAE, self).__init__()
        

        
        self.z_dimension = dim
#        self.register_backward_hook(grad_hook)
        self.fc1 = nn.Linear(T2, 400) # stacked MNIST to 400
        self.fc21 = nn.Linear(400, self.z_dimension) # two hidden low D
        self.fc22 = nn.Linear(400, self.z_dimension) # layers, same size
        self.fc3 = nn.Linear(self.z_dimension, 400)  
        self.fc4 = nn.Linear(400, T2)

    def encode(self, x): 
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) # <- Stochasticity!!!
        # How can the previous line allow back propagation?
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, T2))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def generate(self):  
        pass
        #  Need something like z = linspace(-sigma,sigma,...), then self.decode(z)
        #   Want to raster scan the decode function via its inputs, rather than 
        #    sampling randomly. 
        #   
    
    def get_samples(self, mu, logvar):
        Tmu = torch.tensor(mu, dtype=torch.float).to(device)
        Tlogvar = torch.tensor(logvar, dtype=torch.float).to(device)
        z = self.reparameterize(Tmu, Tlogvar)
        sample = model.decode(z)
        return sample

    def average_all_mu(self,test=False):
        mu_all = torch.tensor(np.zeros((10,self.z_dimension)),dtype=torch.float)
        
        if not test:      
            for batch_idx, (data, which_digit) in enumerate(train_loader):
                if batch_idx > 467: #last batch only has 96 examples (#468)
                    break
                data.to(device)
                
                fc21current,fc22current = model.encode(data.cuda().view(-1,T2))
        #        print('fc21',fc21current.size())
                for i in range(10):
                    this_digit = which_digit == i
        #            print('fc21',fc21current.size())
        #            print(mu_all.size(),this_digit.size())
        
                    mu_all[i,:] = \
                    torch.sum(fc21current[this_digit,:],0)
                    mu_count = torch.sum(this_digit,0)
            mu_all = mu_all/mu_count
            self.mu0 = mu_all
        else:
            for batch_idx, (data, which_digit) in enumerate(test_loader):
                if batch_idx > 467: #last batch only has 96 examples (#468)
                    break
                data.to(device)
                
                fc21current,fc22current = model.encode(data.cuda().view(-1,T2))
        #        print('fc21',fc21current.size())
                for i in range(10):
                    this_digit = which_digit == i
        #            print('fc21',fc21current.size())
        #            print(mu_all.size(),this_digit.size())
        
                    mu_all[i,:] = \
                    torch.sum(fc21current[this_digit,:],0)
                    mu_count = torch.sum(this_digit,0)

            mu_all = mu_all/mu_count
            self.mu_test = mu_all
        return 
        


if 'model' not in locals():
    print('new randomly initialized model...\n')
    model = VAE().to(device)

if False: # F9 this to start with a trained model. 
    model.load_state_dict(torch.load('VAEresults1_VAE20190808_1155')) # KTO favorite
    model.load_state_dict(torch.load('VAE20190716_1551')) # WJP favorite

optimizer = optim.Adam(model.parameters(), lr=1e-5)
beta = 0.5

#corpus = 

#this is a a way to abbreviate some steps
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, beta):
    
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, T2), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD

 
def train(epoch):
    model.train()
    train_loss = 0


    for batch_idx, data in enumerate(train_loader):
        if batch_idx > 467: #last bactch only has 96 examples (#468)
            break
        data = data.type(torch.float)/255.0
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, beta)

        loss.backward()

        train_loss += loss.item()
        optimizer.step()
#
#




#        if batch_idx % args.log_interval == 0:
#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                epoch, batch_idx * len(data), len(train_loader.dataset),
#                100. * batch_idx / len(train_loader),
#                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


#    torch.save(model.state_dict(),'VAETissues' + str(epoch)+'_VAE' + date_for_filename())
    
##    LoadData(epoch)
#
#    display_images(recon_batch)

            

    
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)

            test_loss += loss_function(recon_batch, data, mu, logvar, beta).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
#                save_image(comparison.cpu(),
#                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)


    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))

def date_for_filename():
    tgt = time.localtime()
    year = str(tgt.tm_year)
    mon = "{:02}".format(tgt.tm_mon)
    day = "{:02}".format(tgt.tm_mday)
    hour = "{:02}".format(tgt.tm_hour)
    minute = "{:02}".format(tgt.tm_min)
    datestr = year + mon + day + '_' + hour + minute
    return datestr




if __name__ == "__main__":
    if "cosine_sim" not in locals():
        cosine_sim=plt.figure()
#    if "fc2fig" not in locals():
#        fc2fig, fc2axes = plt.subplots(2,1)
#
#    if "fc4fig" not in locals():
#        fc4fig = plt.figure()
# 
#    if "hist_fig" not in locals():
#        hist_fig, hist_axes = plt.subplots(3,4)
#
#    if "corr_fig" not in locals():
#        corr_fig, corr_axes = plt.subplots(1,1)
#        
#    if "con_fig" not in locals():
#        con_fig=plt.figure()
#        
#    if "num_mean" not in locals():
#        num_mean=plt.figure()
#     
#    if "cosine_sim" not in locals():
#        cosine_sim=plt.figure()
#    
#   fig,ax = plt.subplots(3,3)
    
   
    
    for epoch in range(1, args.epochs + 1):
        train(epoch)

#        display_bottleneck(fc2axes)
#        plt.figure(fc4fig.number)
#        display_images(ACQUIRED_DATA)
#        
#        display_as_histogram(hist_axes)
#        
#        plt.figure(fc2fig.number)
#        display_bottleneck(fc2axes)
#        
#        plt.figure(fc4fig.number)
#        display_images(ACQUIRED_DATA)
#        
#        plt.figure(corr_fig.number)
#        display_corr()
#        
#        plt.figure(con_fig.number)
#        display_relationship_vector()
#        
#        plt.figure(num_mean.number)
#        display_means_relationship()
#        
#        plt.figure(cosine_sim.number)
#        cosine_similiarity()
        
#        test(epoch)
#        model.average_all_mu(test=True)
        
#        for i in range(3):
#            for j in range(3):
#                this_digit = i*3+j+1
#                img=model.decode(model.mu_test[this_digit,:].to(device)).cpu().detach().numpy().reshape((28,28))
#                ax[i,j].imshow(img)
#                ax[i,j].set_title(str(this_digit))
#        plt.pause(0.5)
#        with torch.no_grad():
#            sample = torch.randn(64, 20).to(device)
#            sample = model.decode(sample).cpu()
#            save_image(sample.view(64, 1, 28, 28),
#                       'VAEresults/sample_' + str(epoch) + '.png')