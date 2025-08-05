import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import deepxde as dde
import siren_pytorch as siren
from torch.utils.data import Dataset, DataLoader
# import openturns as ot

import numpy as np
import copy
import os

# partial linear model 
# y = x beta + f(t) + epsilon


class Data(Dataset):
    def __init__(self,beta,n,sigma,mode=1,func_f=None,seed=20230915,device='cpu'):
        super().__init__()
        # beta: 2*1
        self.beta = beta
        self.n = n
        self.sigma = sigma
        if func_f is None:
            self.func_f = lambda t_: (t_[:,0]**2)*(t_[:,1]**3) + np.log(1+t_[:,2]) + np.sqrt(1+t_[:,3]*t_[:,4]) + np.exp(t_[:,4]/2)
        else:
            self.func_f = func_f
        np.random.seed(seed)
        self.d = 6
        self.device = device
            
    def generate(self,path):
        if os.path.exists(path):
            data = np.load(path)
            self.data = data['y']
            self.x = data['x']
            self.t_ = data['t']
            # self.f = data['true_f']
            self.len = self.data.shape[0]
        else:
            pass

    def __getitem__(self, index):
        return {'y':torch.tensor(self.data[index,:]).float().to(self.device),
                'x':torch.tensor(self.x[index,:]).float().to(self.device),
                't':torch.tensor(self.t_[index,:]).float().to(self.device)
                }
    
    def __len__(self):
        return self.len
    
        
class Net(nn.Module):
    def __init__(self, d, seed=20230915, layer_mat=None,act=None):
        super().__init__()
        if layer_mat is None:
            self.layer_mat = [2,1000,1000,1]
        else:
            self.layer_mat = layer_mat
        if act is None:
            act = 'ReLU'
        self.layer_num = len(layer_mat) - 1
        self.net = nn.Sequential()
        self.net.add_module(str(0) + "linear", nn.Linear(layer_mat[0], layer_mat[1],bias=True))
        for i in range(self.layer_num - 1):
            if act == 'ReLU':
                self.net.add_module(str(i) + "Act", nn.ReLU())
            if act == 'Tanh':
                self.net.add_module(str(i) + "Act", nn.Tanh())
            if act == 'Sine':
                self.net.add_module(str(i) + "Act", siren.Sine())
            #self.net.add_module(str(i) + 'Dropout', nn.Dropout(p=0.9))
            self.net.add_module(str(i+1) + "linear",
                                nn.Linear(layer_mat[i+1], layer_mat[i+2],bias=True))
        self._random_seed(seed)
        self._initial_param()
        self.beta =  nn.Parameter(torch.zeros((2,1)).float())

    def forward(self, grid_point):
        return self.net(grid_point)

    def _initial_param(self):
        for name, param in self.net.named_parameters():
            # using sqrt(gain/input_dim)N(0,1) as initial parameters for weights and the first layer bias
            if name.endswith('weight'):
                # nn.init.kaiming_normal_(param,mode='fan_out', nonlinearity='relu')
                nn.init.normal_(param, mean=0.0, std=(2/self.layer_mat[1])**0.5)
            elif name.startswith(str(0)) and name.endswith('bias'):
                nn.init.normal_(param, mean=0.0, std=(2/self.layer_mat[1])**0.5)
            else:
                nn.init.zeros_(param)
                
    def _random_seed(self,seed):
        torch.manual_seed(seed)
        
        
    
        
class Under_Net(nn.Module):
    def __init__(self, d, seed=20230915, layer_mat=None,act=None):
        super().__init__()
        if layer_mat is None:
            self.layer_mat = [2,20,20,20,1]
        else:
            self.layer_mat = layer_mat
        if act is None:
            act = 'ReLU'
        self.layer_num = len(layer_mat) - 1
        self.net = nn.Sequential()
        self.net.add_module(str(0) + "linear", nn.Linear(layer_mat[0], layer_mat[1],bias=True))
        for i in range(self.layer_num - 1):
            if act == 'ReLU':
                self.net.add_module(str(i) + "Act", nn.ReLU())
            if act == 'Tanh':
                self.net.add_module(str(i) + "Act", nn.Tanh())
            if act == 'Sine':
                self.net.add_module(str(i) + "Act", siren.Sine())
            #self.net.add_module(str(i) + 'Dropout', nn.Dropout(p=0.9))
            self.net.add_module(str(i+1) + "linear",
                                  nn.Linear(layer_mat[i+1], layer_mat[i+2],bias=True))
        self._random_seed(seed)
        self._initial_param()
        self.beta =  nn.Parameter(torch.zeros((2,1)).float())

    def forward(self, grid_point):
        return self.net(grid_point)

    def _initial_param(self):
        for name, param in self.net.named_parameters():
            # using sqrt(gain/input_dim)N(0,1) as initial parameters for weights and the first layer bias
            if name.endswith('weight'):
                nn.init.kaiming_normal_(param,mode='fan_out', nonlinearity='relu')
            else:
                nn.init.zeros_(param)
                
    def _random_seed(self,seed):
        torch.manual_seed(seed)