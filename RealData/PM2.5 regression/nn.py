import os

import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--n','-n',default = 1435, type = int, help='sample size')
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
    path_checkpoint = "./checkpoint/partial_linear_nn_%d_%d.pth"%(n,m)
    path_train_data = "./data/partial_linear_train.npz"
    path_validation_data = "./data/partial_linear_validation.npz"
    path_test_data = "./data/partial_linear_test.npz"
    
    d = 6
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
            loss = criterion(y,y_hat)
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
                loss = criterion(y,y_hat)
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
    criterion = nn.MSELoss()
    #optimizer = optim.Adam(under_net.parameters(), lr=args.lr, betas=(0.9,0.999))
    optimizer = optim.SGD(under_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch)
    
    n_validation = 287
    n_test = 1434
    # print('==> Preparing data..')
    trainset = model.Data(beta,round(0.8*n),sigma,device=device)
    # trainset.generate(path_train,over_net)
    trainset.generate(path_train_data)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size = n//10, shuffle=True, num_workers=0,generator=torch.Generator(device))
    # print('Train data loaded')
    validationset = model.Data(beta,n_validation,sigma,device=device)
    # validationset.generate(path_validation,over_net)
    validationset.generate(path_validation_data)
    validationloader = torch.utils.data.DataLoader(
        validationset, batch_size=n_validation//10, shuffle=True, num_workers=0,generator=torch.Generator(device))

    testset = model.Data(beta,n_test,sigma,device=device)
    # validationset.generate(path_validation,over_net)
    testset.generate(path_test_data)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=n_test//10, shuffle=True, num_workers=0,generator=torch.Generator(device))
    # print('Validation data loaded')

    
    last_loss = torch.inf
    for epoch in range(start_epoch, start_epoch+total_epoch):
        if epoch % 200 == 0:
            print('seed %d, epoch %d already done!'%(seed,epoch))
        train_loss = train(epoch)
        if epoch % 20 == 0:
            last_loss = validation(epoch,path_checkpoint,last_loss)
        scheduler.step()
    # x_centered = torch.mean(validationset.x,dim = 0,keepdim=False)
    # Sigma_hat = torch.zeros((2,2))
    # with torch.no_grad():
    #     for batch_idx, states in enumerate(validationloader):
    #         x = states['x']
    #         t_ = states['t']
    #         y = states['y']
    #         y_hat = under_net(t_)  + torch.matmul(x,under_net.beta) 
    #         Sigma_hat = Sigma_hat + torch.sum(torch.einsum('ij,isl->isl',torch.square(y - y_hat),torch.einsum('ij,ik->ijk',x - x_centered,x - x_centered)),dim=0,keepdims=False)
    # Sigma_hat = Sigma_hat / validationset.__len__()
    # beta = torch.tensor([[1.],[.75]],dtype=torch.float)
    # if torch.matmul(torch.transpose(under_net.beta - beta,1,0),torch.linalg.solve(Sigma_hat,under_net.beta-beta)) > 5.991:
    #     size = 1
    # else:
    #     size = 0
    test_f_loss = 0
    with torch.no_grad():
        for batch_idx, states in enumerate(testloader):
            x = states['x']
            t_ = states['t']
            y = states['y']
            y_hat = under_net(t_) + torch.matmul(x,under_net.beta)
            loss = criterion(y,y_hat)
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
        
    # path_train = "./data/partial_linear_train_%d_%.3f_%d.npz"%(n,sigma,mode)
    # path_validation = "./data/partial_linear_validation_%d_%.3f_%d.npz"%(n,sigma,mode)
    # path_checkpoint = "./checkpoint/partial_linear_test_%d_%.3f_%.3f_%d_%d.pth"%(n,lamda,sigma,m,mode)

    # processes_repeat = []
    begin = args.begin
    # torch.multiprocessing.set_start_method('spawn')
    for seed in range(begin,1+begin):
        print(seed)
        # processes_repeat = multiprocessing.Process(target=run,args=(seed,n,sigma,lamda,lr,device,m,mode))
        # processes_repeat.start()
        run(seed,n,sigma,lr,device,m,mode)