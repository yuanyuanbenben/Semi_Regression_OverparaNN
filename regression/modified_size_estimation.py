# our method

import numpy as np
import torch
mode = 9
sigma = 0.5
n = 500
m = 1000
para_value = np.ndarray((500,2))
nonpara_value = np.ndarray((500))
size_0 = 0
size_1 = 0
size_2 = 0
size_0_val = 0
size_1_val = 0
size_2_val = 0
for seed in range(500):
    validation_loss_before = np.inf
    for lamda in [1]:
        _checkpoint = torch.load("./checkpoint/partial_linear_main_%d_%.3f_%.3f_%d_%d_%d.pth"%(n,lamda,sigma,m,mode,seed))
        if _checkpoint['validation_loss'] < validation_loss_before:
            checkpoint = _checkpoint
            validation_loss_before = checkpoint['validation_loss']
            _lamda = lamda
    # print(_lamda)
    para_value[seed,:] = torch.Tensor.cpu(checkpoint['beta'].data).numpy().reshape(-1)
    nonpara_value[seed] = checkpoint['test_f_loss']
    sigma_mat = np.array([[checkpoint['lbeta_0'],checkpoint['lbeta_01']],[checkpoint['lbeta_01'],checkpoint['lbeta_1']]])
    sigma_inverse = np.linalg.inv(sigma_mat)
    sigma_esti_0 = np.sqrt(checkpoint['train_loss'] * sigma_inverse[0,0]/ 0.8/n) * 1.96
    sigma_esti_1 = np.sqrt(checkpoint['train_loss'] * sigma_inverse[1,1]/ 0.8/n) * 1.96
    if abs(para_value[seed,0] - 1) > sigma_esti_0:
        size_0 = size_0 + 1
    if abs(para_value[seed,1] - 0.75) > sigma_esti_1:
        size_1 = size_1 + 1
    sigma_esti_0 = np.sqrt(checkpoint['validation_loss'] * sigma_inverse[0,0]/ 0.8/n) * 1.96
    sigma_esti_1 = np.sqrt(checkpoint['validation_loss'] * sigma_inverse[1,1]/ 0.8/n) * 1.96
    if abs(para_value[seed,0] - 1) > sigma_esti_0:
        size_0_val = size_0_val + 1
    if abs(para_value[seed,1] - 0.75) > sigma_esti_1:
        size_1_val = size_1_val + 1

bar_beta_ = np.mean(para_value,axis=0)
bias = bar_beta_ - [1,0.75]
print('bias:%.6f'%(np.sqrt(np.sum(np.square(bias)))))
std_beta = np.sum(np.mean(np.square(para_value - bar_beta_),axis=0))**0.5
print('std:%.6f'%(std_beta))
print('mse:%.6f'%(np.sqrt(np.sum(np.square(bias)))**2 + std_beta**2))
mse_f = np.mean(nonpara_value,axis = 0)
print('mse:%.6f'%(mse_f))
print(size_0/500)
print(size_1/500)
# print(size_0_val/500)
# print(size_1_val/500)