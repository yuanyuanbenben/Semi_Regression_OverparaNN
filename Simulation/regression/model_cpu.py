import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import deepxde as dde
import siren_pytorch as siren
from torch.utils.data import Dataset, DataLoader
import openturns as ot

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
            self.func_f = lambda t_: 2 * np.sin(t_[:,0]) * np.cos(t_[:,1]+t_[:,4]**2) + np.sin(t_[:,2]+t_[:,3]**2) + np.sqrt(t_[:,1]+t_[:,4]+t_[:,2]*t_[:,4])
        else:
            self.func_f = func_f
        np.random.seed(seed)
        if (mode == 0) or (mode == 5):
            self.d = 1
        elif (mode == 1) or (mode == 3):
            self.d = 5
        elif (mode == 2) or (mode == 4):
            self.d = 3
        self.device = device
            
    def generate(self,path,init_f=None):
            
        if os.path.exists(path):
            data = np.load(path)
            self.data = data['y']
            self.x = data['x']
            self.t_ = data['t']
            self.len = self.data.shape[0]
        else:
            self.x = np.random.uniform(0,1,(self.n,2))
            self.t_ = np.random.uniform(0,1,(self.n,self.d))
            self.data = np.matmul(self.x,self.beta) + self.func_f(self.t_).reshape(self.n,-1) + np.random.normal(scale=self.sigma,size=(self.n,1))
            self.len = self.n
            np.savez(path,x=self.x,y=self.data,t=self.t_)
        if init_f is None:
            self.init_value = None
        else:
            with torch.no_grad():
                self.init_value = init_f(torch.tensor((self.t_)).float().to(self.device))
                
    def _generate(self,init_f=None):
        # self.x = np.random.uniform(0,1,(self.n,2))
        # self.x[:,1] = self.x[:,0]/2 + self.x[:,1]/2
        # self.t_ = np.random.uniform(0,1,(self.n,self.d))
        # if self.d > 1:
        #     for i in range(self.d - 1):
        #         self.t_[:,i + 1] = self.t_[:,i]/2 + self.t_[:,i + 1]/2
                
        Rx = ot.CorrelationMatrix(2)
        Rx[0, 1] = 0.5
        copula_x = ot.NormalCopula(Rx)
        self.x = np.array(copula_x.getSample(self.n))
        
        Rt = ot.CorrelationMatrix(self.d)
        for i in range(self.d-1):
            Rt[i,i+1] = 0.5
        copula_t = ot.NormalCopula(Rt)
        self.t_ =  np.array(copula_t.getSample(self.n))
        self.data = np.matmul(self.x,self.beta) + self.func_f(self.t_).reshape(self.n,-1) + np.random.normal(scale=self.sigma,size=(self.n,1))
        self.len = self.n
        if init_f is None:
            self.init_value = None
        else:
            with torch.no_grad():
                self.init_value = init_f(torch.tensor((self.t_)).float().to(self.device))
        
    def __getitem__(self, index):
        if self.init_value is None:
            return {'y':torch.tensor(self.data[index,:]).float(),
                    'x':torch.tensor(self.x[index,:]).float(),
                    't':torch.tensor(self.t_[index,:]).float()}
        else:
            return {'y':torch.tensor(self.data[index,:]).float().to(self.device)+self.init_value[index,:],
                    'x':torch.tensor(self.x[index,:]).float(),
                    't':torch.tensor(self.t_[index,:]).float(),}
    
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
                nn.init.kaiming_normal_(param,mode='fan_out', nonlinearity='relu')
            elif name.startswith(str(0)) and name.endswith('bias'):
                nn.init.normal_(param, mean=0.0, std=(2/self.layer_mat[1])**0.5)
            else:
                nn.init.zeros_(param)
                
    def _random_seed(self,seed):
        torch.manual_seed(seed)