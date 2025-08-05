import numpy as np
import torch
mode = 0
sigma = 0.5
n = 500
m = 1000
para_value = np.ndarray((200,2))
nonpara_value = np.ndarray((200))
size_0 = 0
size_1 = 0
size_2 = 0
size_0_val = 0
size_1_val = 0
size_2_val = 0
for seed in range(200):
    validation_loss_before = np.inf
    for lamda in [1]:
        _checkpoint = torch.load("./checkpoint/partial_linear_main_%d_%.3f_%.3f_%d_%d_%d.pth"%(n,lamda,sigma,m,mode,seed))
        # if _checkpoint['validation_loss'] < validation_loss_before:
        checkpoint = _checkpoint
        # validation_loss_before = checkpoint['validation_loss']
        _lamda = lamda
    # print(_lamda)
    para_value[seed,:] = torch.Tensor.cpu(checkpoint['beta'].data).numpy().reshape(-1)
    nonpara_value[seed] = checkpoint['test_f_loss']
    size_0 += checkpoint['size_0']
    size_1 += checkpoint['size_1']
    size_2 += checkpoint['size_2']
bar_beta_ = np.mean(para_value,axis=0)
bias = bar_beta_ - [1,0.75]
print('bias:%.6f'%(np.sqrt(np.sum(np.square(bias)))))
std_beta = np.sum(np.mean(np.square(para_value - bar_beta_),axis=0))**0.5
print('std:%.6f'%(std_beta))
print('mse:%.6f'%(np.sqrt(np.sum(np.square(bias)))**2 + std_beta**2))
mse_f = np.mean(nonpara_value,axis = 0)
print('mse:%.6f'%(mse_f))
print(size_0/200)
print(size_1/200)
print(size_2/200)
size_0 = 0
size_1 = 0
size_2 = 0
for seed in range(500):
    validation_loss_before = np.inf
    for lamda in [1]:
        _checkpoint = torch.load("./checkpoint/partial_linear_main_%d_%.3f_%.3f_%d_%d_%d.pth"%(n,lamda,sigma,m,mode,seed))
        # if _checkpoint['validation_loss'] < validation_loss_before:
        checkpoint = _checkpoint
        # validation_loss_before = checkpoint['validation_loss']
        _lamda = lamda
    size_0 += checkpoint['size_0']
    size_1 += checkpoint['size_1']
    size_2 += checkpoint['size_2']
print(1-size_0/500)
print(1-size_1/500)
print(1-size_2/500)

para_value = np.ndarray((200,2))
nonpara_value = np.ndarray((200))
for seed in range(200):
    checkpoint = torch.load("./checkpoint/partial_linear_nn_%d_%.3f_%d_%d_%d.pth"%(n,sigma,8,mode,seed))
    para_value[seed,:] = torch.Tensor.cpu(checkpoint['beta'].data).numpy().reshape(-1)
    nonpara_value[seed] = checkpoint['test_f_loss']
bar_beta_ = np.mean(para_value,axis=0)
bias = bar_beta_ - [1,0.75]
print('bias:%.6f'%(np.sqrt(np.sum(np.square(bias)))))
std_beta = np.sum(np.mean(np.square(para_value - bar_beta_),axis=0))**0.5
print('std:%.6f'%(std_beta))
print('mse:%.6f'%(np.sqrt(np.sum(np.square(bias)))**2 + std_beta**2))
mse_f = np.mean(nonpara_value,axis = 0)
print('mse:%.6f'%(mse_f))

para_value = np.ndarray((200,2))
nonpara_value = np.ndarray((200))
for seed in range(200):
    checkpoint = np.load("./output/partial_linear_rkhs_%d_%.3f_%d_%d.npz"%(n,sigma,mode,seed))
    para_value[seed,:] = checkpoint['beta']
    nonpara_value[seed] = checkpoint['non_loss']
bar_beta_ = np.mean(para_value,axis=0)
bias = bar_beta_ - [1,0.75]
print('bias:%.6f'%(np.sqrt(np.sum(np.square(bias)))))
std_beta = np.sum(np.mean(np.square(para_value - bar_beta_),axis=0))**0.5
print('std:%.6f'%(std_beta))
print('mse:%.6f'%(np.sqrt(np.sum(np.square(bias)))**2 + std_beta**2))
mse_f = np.mean(nonpara_value,axis = 0)
print('mse:%.6f'%(mse_f))


para_value = np.ndarray((200,2))
nonpara_value = np.ndarray((200))
for seed in range(200):
    checkpoint = np.load("./output/partial_linear_locallinear_%d_%.3f_%d_%d.npz"%(n,sigma,mode,seed))
    para_value[seed,:] = checkpoint['beta']
    nonpara_value[seed] = checkpoint['non_loss']
bar_beta_ = np.mean(para_value,axis=0)
bias = bar_beta_ - [1,0.75]
print('bias:%.6f'%(np.sqrt(np.sum(np.square(bias)))))
std_beta = np.sum(np.mean(np.square(para_value - bar_beta_),axis=0))**0.5
print('std:%.6f'%(std_beta))
print('mse:%.6f'%(np.sqrt(np.sum(np.square(bias)))**2 + std_beta**2))
mse_f = np.mean(nonpara_value,axis = 0)
print('mse:%.6f'%(mse_f))

para_value = np.ndarray((200,2))
nonpara_value = np.ndarray((200))
for seed in range(200):
    checkpoint = np.load("./output/partial_linear_spline_%d_%.3f_%d_%d.npz"%(n,sigma,mode,seed))
    para_value[seed,:] = checkpoint['beta']
    nonpara_value[seed] = checkpoint['non_loss']
bar_beta_ = np.mean(para_value,axis=0)
bias = bar_beta_ - [1,0.75]
print('bias:%.6f'%(np.sqrt(np.sum(np.square(bias)))))
std_beta = np.sum(np.mean(np.square(para_value - bar_beta_),axis=0))**0.5
print('std:%.6f'%(std_beta))
print('mse:%.6f'%(np.sqrt(np.sum(np.square(bias)))**2 + std_beta**2))
mse_f = np.mean(nonpara_value,axis = 0)
print('mse:%.6f'%(mse_f))
