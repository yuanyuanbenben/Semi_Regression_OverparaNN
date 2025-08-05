import os

import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--n','-n',default = 500, type = int, help='sample size')
parser.add_argument('--sigma','-sigma','-s',default = 0.2, type = float, help='variance of noise')
parser.add_argument('--width','-w',default = 6, type = int, help='width of neural network')
parser.add_argument('--gpu_ids','-cuda','-c',default = '0', type = str, help='cuda device')
parser.add_argument('--mode','-m',default = 0, type = int, help='mode')
parser.add_argument('--begin','-seed',default = 0, type = int, help='start epoch')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids


import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
# import deepxde as dde
# import siren_pytorch as siren
# from torch.utils.data import Dataset, DataLoader
# import multiprocessing

import numpy as np
import copy

import model 

# from utils import progress_bar, TemporaryGrad


def run(seed,n,sigma,lr,device,m,mode):
    torch.manual_seed(20230718) 
    path_checkpoint = "./checkpoint/partial_linear_nn_%d_%.3f_%d_%d_%d.pth"%(n,sigma,m,mode,seed)
    path_train_data = "./data/partial_linear_train_%d_%.3f_%d_%d.npz"%(n,sigma,mode,seed)
    path_validation_data = "./data/partial_linear_validation_%d_%.3f_%d_%d.npz"%(n,sigma,mode,seed)
    path_test_data = "./data/partial_linear_test_%d_%.3f_%d_%d.npz"%(n,sigma,mode,seed)
    if mode == 0:
        d = 5
        func_f = lambda t_: ((t_[:,0]**2)*(t_[:,1]**3) + np.log(1+t_[:,2]) + np.sqrt(1+t_[:,3]*t_[:,4]) + np.exp(t_[:,4]/2))*5 - 15.25

    if mode == 1:
        d = 10
        func_f = lambda t_: (((t_[:,0]**2)*(t_[:,1]**3) + np.log(1+t_[:,2]) + np.sqrt(1+t_[:,3]*t_[:,4]) + np.exp(t_[:,4]/2) + 
                            (t_[:,5]**2)*(t_[:,6]**3)/2 + np.log(1+t_[:,7])*2 + np.sqrt(1+t_[:,8]*t_[:,9])*2 + np.exp(t_[:,9]/2)/2))*5/2 - 17.25

    if mode == 8:
        d = 5
        def func_f(t_):
            ret = np.zeros_like(t_[:,0])
            for j in range(d):
                ret = ret + (j+1)*t_[:,j]
            ret = np.sin(6*np.pi/d/(d+1)*ret)*5 + 1
            return ret
        
    if mode == 9:
        d = 10
        def func_f(t_):
            ret = np.zeros_like(t_[:,0])
            for j in range(d):
                ret = ret + (j+1)*t_[:,j]
            ret = np.sin(6*np.pi/d/(d+1)*ret)*5 + 2.3
            return ret

    beta = np.array([[1],[0.75]])
    
    def train(epoch):
        # print('\nEpoch: %d' % epoch)
        under_net.train()
        train_loss = 0
        for batch_idx, states in enumerate(trainloader):
            optimizer.zero_grad()
            x = states['x']
            t_ = states['t']
            y = states['y']
            y_hat = under_net(t_) + torch.matmul(x,under_net.beta)
            loss = criterion(y_hat,y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # progress_bar(batch_idx, len(trainloader), 'Average Loss: %.3f | Current Loss: %.3f'
            #             % (train_loss/(batch_idx+1), loss.item()))
        return train_loss/(batch_idx+1)


    def validation(epoch,path_checkpoint,last_loss):
        # print('\nEpoch: %d validation' % epoch)
        under_net.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch_idx, states in enumerate(validationloader):
                optimizer.zero_grad()
                x = states['x']
                t_ = states['t']
                y = states['y']
                y_hat = under_net(t_)  + torch.matmul(x,under_net.beta)
                loss = criterion(y_hat,y)
                validation_loss += loss.item()
                    
        #         progress_bar(batch_idx, len(validationloader), 'Average Loss: %.3f | Current Loss: %.3f'
        #                 % (validation_loss/(batch_idx+1), loss.item()))
        print('#########################')
        print(under_net.beta)
        print('#########################')
        validation_loss = validation_loss/(batch_idx+1)
        # if validation_loss < last_loss and epoch > 3000 :
        #     last_loss = validation_loss
            #print('Saving..')
            
            
        return validation_loss
    
    # Model
    print('==> Building model..')
    under_net = model.Under_Net(d,layer_mat=[d,m,m,m,m,1])
    
    under_net = under_net.to(device)
    # if device == 'cuda':
    #     under_net.net = torch.nn.DataParallel(under_net.net)
    #     cudnn.benchmark = True
   
    
    start_epoch = 0
    # initial_weight = get_w0()
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(path_checkpoint)
        under_net.load_state_dict(checkpoint['under_net'])
        start_epoch = checkpoint['epoch']
        under_net.beta = checkpoint['beta']
    
    total_epoch = 1001
    # criterion = nn.MSELoss()
    criterion = nn.BCEWithLogitsLoss()
    criterion_weight = nn.MSELoss()
    #optimizer = optim.Adam(under_net.parameters(), lr=args.lr, betas=(0.9,0.999))
    optimizer = optim.SGD(under_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch)
    
    # Data            
    # print('==> Preparing data..')
    trainset = model.Data(beta,round(0.8*n),sigma,mode=mode,func_f=func_f,seed = 20230915+seed,device=device)
    # trainset.generate(path_train,over_net)
    trainset.generate(path_train_data)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size = n//10, shuffle=True, num_workers=0,generator=torch.Generator(device))
    # print('Train data loaded')
    validationset = model.Data(beta,n-round(0.8*n),sigma,mode=mode,func_f=func_f,seed = 20230915+seed+10000,device=device)
    # validationset.generate(path_validation,over_net)
    validationset.generate(path_validation_data)
    validationloader = torch.utils.data.DataLoader(
        validationset, batch_size=n//10, shuffle=True, num_workers=0,generator=torch.Generator(device))

    testset = model.Data(beta,n,sigma,mode=mode,func_f=func_f,seed = 20230915+seed+20000,device=device)
    # validationset.generate(path_validation,over_net)
    testset.generate(path_test_data)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=n//10, shuffle=True, num_workers=0,generator=torch.Generator(device))
    # print('Validation data loaded')

    
    last_loss = torch.inf
    for epoch in range(start_epoch, start_epoch+total_epoch):
        if epoch % 200 == 0:
            print('seed %d, epoch %d already done!'%(seed,epoch))
        train_loss = train(epoch)
        if epoch % 20 == 0:
            last_loss = validation(epoch,path_checkpoint,last_loss)
        scheduler.step()

    test_f_loss = 0
    with torch.no_grad():
        for batch_idx, states in enumerate(testloader):
            t_ = states['t']
            true_f = states['true_f']
            f_hat = under_net(t_)
            loss = criterion_weight(true_f,f_hat)
            # for _, param in over_net.named_parameters():
            #     if _.endswith('weight'):
            #         loss = loss + criterion(param,initial_weight[_]) * lamda 
            test_f_loss += loss.item()
    test_f_loss = test_f_loss/(batch_idx+1)
    print('test_f_loss',test_f_loss)
    state = {
        'under_net': under_net.state_dict(),
        'train_loss': train_loss,
        'validation_loss': last_loss,
        'epoch': epoch,
        'seed':torch.initial_seed(),
        'beta':under_net.beta,
        'test_f_loss':test_f_loss,
        # 'size':size
    }
    torch.save(state,path_checkpoint)


if __name__ == '__main__':
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # parameters setting
    
    n = args.n
    sigma = args.sigma
    lr = args.lr
    m = args.width

    mode = args.mode


    if not os.path.isdir('data'):
        os.mkdir('data')
    if not os.path.isdir('output'):
        os.mkdir('output')
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
        
    begin = args.begin
    for seed in range(begin,50+begin):
        print(seed)
        run(seed,n,sigma,lr,device,m,mode)