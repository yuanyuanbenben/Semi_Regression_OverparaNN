'''   simulation for partial linear model  '''
import os

import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--n','-n',default = 500, type = int, help='sample size')
parser.add_argument('--sigma','-sigma','-s',default = 0.2, type = float, help='variance of noise')
parser.add_argument('--lamda','-lamda','-l',default = 1000000, type = float, help='tuning parameters')
parser.add_argument('--gpu_ids','-cuda','-c',default = '0', type = str, help='cuda device')
parser.add_argument('--mode','-m',default = 0, type = int, help='mode')
parser.add_argument('--begin','-seed',default = 0, type = int, help='start epoch')
parser.add_argument('--width','-w',default = 1000, type = int, help='width of neural network')
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



def run(seed,n,sigma,lamda,lr,device,m,mode):
    torch.manual_seed(20230718) 
    path_checkpoint = "./checkpoint/partial_linear_main_%d_%.3f_%.3f_%d_%d_%d.pth"%(n,lamda,sigma,m,mode,seed)
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
    
    def adjust_learning_rate(optimizer):
        lr = args.lr * 3
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def get_w0(initial_print=True):
        # if initial_print:
        #     print('==> Get initial weight matrix...')
        direct_proj_dic = {}    
        for _, param in over_net.named_parameters():
            direct_proj_dic[_] = copy.deepcopy(param)
        return direct_proj_dic
        
    def train(epoch,lamda,initial_weight):
        # print('\nEpoch: %d' % epoch)
        over_net.train()
        train_loss = 0
        # phi_loss = 0
        for batch_idx, states in enumerate(trainloader):
            optimizer.zero_grad()
            x = states['x']
            t_ = states['t']
            y = states['y']
            with torch.no_grad():
                f_init = over_net_initial(t_)
            y_hat = over_net(t_) - f_init + torch.matmul(x,over_net.beta)
            loss = criterion(y_hat,y)
            for _, param in over_net.named_parameters():
                if _.endswith('weight'):
                    loss = loss + criterion_weight(param,initial_weight[_]) * lamda 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # with torch.no_grad():
            #     phi_loss += torch.mean(torch.square(1+torch.exp(-y_hat))*torch.exp(y_hat)).item()

            
            # progress_bar(batch_idx, len(trainloader), 'Average Loss: %.3f | Current Loss: %.3f'
            #             % (train_loss/(batch_idx+1), loss.item()))
        return train_loss/(batch_idx+1)


    def validation(epoch,path_checkpoint,initial_weight,last_loss):
        # print('\nEpoch: %d validation' % epoch)
        over_net.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch_idx, states in enumerate(validationloader):
                optimizer.zero_grad()
                x = states['x']
                t_ = states['t']
                y = states['y']
                f_init = over_net_initial(t_)
                y_hat = over_net(t_) -  f_init + torch.matmul(x,over_net.beta)
                loss = criterion(y_hat,y)
                # for _, param in over_net.named_parameters():
                #     if _.endswith('weight'):
                #         loss = loss + criterion(param,initial_weight[_]) * lamda 
                validation_loss += loss.item()
                    
        #         progress_bar(batch_idx, len(validationloader), 'Average Loss: %.3f | Current Loss: %.3f'
        #                 % (validation_loss/(batch_idx+1), loss.item()))
        print('#########################')
        print(over_net.beta)
        print('#########################')
        validation_loss = validation_loss/(batch_idx+1)
        if epoch == 1000:
            last_loss = validation_loss
        return last_loss
    
    # Model
    print('==> Building model..')
    over_net = model.Net(d,layer_mat=[d,m,m,m,m,1])
    over_net_initial = model.Net(d,layer_mat=[d,m,m,m,m,1])
    
    over_net = over_net.to(device)
    over_net_initial = over_net_initial.to(device)
    # if device == 'cuda':
    #     over_net.net = torch.nn.DataParallel(over_net.net)
    #     cudnn.benchmark = True
    
    initial_weight = get_w0()
    over_net_initial.load_state_dict(get_w0())
    
    start_epoch = 0
    # initial_weight = get_w0()
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(path_checkpoint)
        over_net.load_state_dict(checkpoint['over_net'])
        over_net_initial.load_state_dict(checkpoint['init_net'])
        start_epoch = checkpoint['epoch']
        over_net.beta = checkpoint['beta']
    
    total_epoch = 1001
    # criterion = nn.MSELoss()
    criterion = nn.BCEWithLogitsLoss()
    criterion_weight = nn.MSELoss()
    #optimizer = optim.Adam(over_net.parameters(), lr=args.lr, betas=(0.9,0.999))
    optimizer = optim.SGD(over_net.parameters(), lr=lr)
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
    # Y_mean = np.mean(trainset.data,keepdims=False)
    # Y_var_inverse = 1/(np.mean(np.square(trainset.data-Y_mean)))
    # print(Y_var_inverse)
    for epoch in range(start_epoch, start_epoch+total_epoch):
        # if epoch >= 5000:
        #     adjust_learning_rate(optimizer)
        if epoch % 200 == 0:
            print('seed %d, epoch %d already done!'%(seed,epoch))
        phi_loss = train(epoch,lamda,initial_weight)
        if epoch % 20 == 0:
            last_loss = validation(epoch,path_checkpoint,initial_weight,last_loss)
        if epoch == 1000:
            #  (X - EZ|X)^2
            # net_0 = model.Under_Net(d,layer_mat=[d,32,128,512,512,128,32,1])
            net_0 = model.Under_Net(d,layer_mat=[d,32,128,128,32,1])
            net_0 = net_0.to(device)
            optimizer_0 = optim.SGD(net_0.parameters(), lr=1e-3)
            # criterion_0 = nn.MSELoss()
            def weighted_mse_loss(input, target, weight):
                return (weight * (input - target) ** 2).sum() / weight.sum()
            def phi(x):
                return torch.exp(-x)/torch.square(1+torch.exp(-x))
            criterion_0 = weighted_mse_loss
            loss_old = torch.inf
            train_t = torch.tensor(trainset.t_).float().to(device)
            train_x = torch.tensor(trainset.x).float().to(device)
            validation_t = torch.tensor(validationset.t_).float().to(device)
            validation_x = torch.tensor(validationset.x).float().to(device)
            with torch.no_grad():
                p_hat = over_net(train_t) -  over_net_initial(train_t) + torch.matmul(train_x,over_net.beta)
                weight = phi(p_hat)
                p_hat_validation = over_net(validation_t) -  over_net_initial(validation_t) + torch.matmul(validation_x,over_net.beta)
                weight_validation = phi(p_hat_validation)
            for i in range(5000):
                optimizer_0.zero_grad()
                y_hat = net_0(train_t)
                loss_0 = criterion_0(train_x[:,0:1],y_hat,weight)
                loss_0.backward()
                optimizer_0.step()
                if round(i/10)*10 == i:
                    with torch.no_grad():
                        y_test_hat_0 = net_0(validation_t)
                        test_loss = criterion_0(validation_x[:,0:1],y_test_hat_0,weight_validation).item()
                        if test_loss <= loss_old:
                            # print(i)
                            loss_old = test_loss
                            lbeta_0 = copy.deepcopy(loss_0.item())
                            loss_0 = copy.deepcopy(validation_x[:,0:1] - y_test_hat_0)
            # lbeta[0] = loss.item()
            # net_1 = model.Under_Net(d,layer_mat=[d,32,128,512,512,128,32,1])
            net_1 = model.Under_Net(d,layer_mat=[d,32,128,128,32,1])
            net_1 = net_1.to(device)
            optimizer_1 = optim.SGD(net_1.parameters(), lr=1e-3)
            criterion_1 = weighted_mse_loss
            loss_old = torch.inf
            for i in range(5000):
                optimizer_1.zero_grad()
                y_hat = net_1(train_t)
                loss_1 = criterion_1(train_x[:,1:2],y_hat,weight)
                loss_1.backward()
                optimizer_1.step()
                if round(i/10)*10 == i:
                    with torch.no_grad():
                        y_test_hat_1 = net_1(validation_t)
                        test_loss = criterion_1(validation_x[:,1:2],y_test_hat_1,weight_validation).item()
                        if test_loss <= loss_old:
                            # print(i)
                            loss_old = test_loss
                            lbeta_1 = copy.deepcopy(loss_1.item())
                            loss_1 = copy.deepcopy(validation_x[:,1:2] - y_test_hat_1)
            lbeta_01 = torch.Tensor.cpu(torch.mean(loss_0 * loss_1)).detach().numpy().reshape(-1)
            lbeta_01 = lbeta_01[0]

            sigma_esti_0 = np.sqrt(1 / lbeta_0/ 0.8/n)
            sigma_esti_1 = np.sqrt(1 / lbeta_1/ 0.8/n)
            beta_estimated = torch.Tensor.cpu(over_net.beta.data).numpy().reshape(-1)
            print(beta_estimated[0])
            print(beta_estimated[1])
            print(abs(beta_estimated[0]-beta[0]))
            print(sigma_esti_0 * 1.96)
            print(abs(beta_estimated[1]-beta[1]))
            print(sigma_esti_1 * 1.96)
            if abs(beta_estimated[0]-beta[0]) > sigma_esti_0 * 1.96:
                size_0 = 1
            else: 
                size_0 = 0
            if abs(beta_estimated[1]-beta[1]) > sigma_esti_1 * 1.96:
                size_1 = 1
            else: 
                size_1 = 0
            sigma_mat = np.array([[lbeta_0,lbeta_01],[lbeta_01,lbeta_1]])
            if np.matmul(beta_estimated - beta.reshape(-1), np.matmul(sigma_mat,(beta_estimated - beta.reshape(-1)).T)) > (5.99 / 0.8/n):
                size_2 = 1
            else:
                size_2 = 0
            print(size_0)
            print(size_1)
            print(size_2)
            test_f_loss = 0
            with torch.no_grad():
                for batch_idx, states in enumerate(testloader):
                    t_ = states['t']
                    true_f = states['true_f']
                    f_init = over_net_initial(t_)
                    f_hat = over_net(t_) -  f_init 
                    loss = criterion_weight(true_f,f_hat)
                    # for _, param in over_net.named_parameters():
                    #     if _.endswith('weight'):
                    #         loss = loss + criterion(param,initial_weight[_]) * lamda 
                    test_f_loss += loss.item()
            test_f_loss = test_f_loss/(batch_idx+1)
            print('test_f_loss',test_f_loss)
            state = {
                'over_net': over_net.state_dict(),
                'init_net': initial_weight,
                # 'train_loss': train_loss,
                # 'validation_loss': validation_loss,
                'epoch': epoch,
                'seed':torch.initial_seed(),
                'beta':over_net.beta,
                "lbeta_0":lbeta_0,
                "lbeta_1":lbeta_1,
                "lbeta_01":lbeta_01,
                "size_0":size_0,
                "size_1":size_1,
                'size_2':size_2,
                # "size_0_val":size_0_val,
                # "size_1_val":size_1_val,
                # 'size_2_val':size_2_val,
                'test_f_loss':test_f_loss,
            }
            # print(size)
            torch.save(state,path_checkpoint)

        scheduler.step()


if __name__ == '__main__':
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # parameters setting
    
    n = args.n
    lamda = args.lamda
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
    for seed in range(begin,50+begin):
        print(seed)
        # processes_repeat = multiprocessing.Process(target=run,args=(seed,n,sigma,lamda,lr,device,m,mode))
        # processes_repeat.start()
        run(seed,n,sigma,lamda,lr,device,m,mode)