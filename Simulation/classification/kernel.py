import os
# os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import numpy as np
import copy
import argparse
# from scipy.interpolate import BSpline

import model

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--n','-n',default = 500, type = int, help='sample size')
parser.add_argument('--sigma','-sigma','-s',default = 0.2, type = float, help='variance of noise')
parser.add_argument('--mode','-m',default = 0, type = int, help='mode')
# parser.add_argument('--knots','-k',default = 50, type = int, help='knots of splines')
args = parser.parse_args()

# parameters setting
n = args.n
sigma = args.sigma
# k = args.knots

beta = np.array([[1],[0.75]])


if not os.path.isdir('data'):
    os.mkdir('data')
if not os.path.isdir('output'):
    os.mkdir('output')
    
mode = args.mode

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
    


print('==> Preparing data..')

def gaussian_kernel(t1,t2,h=1):
    return np.exp(-1/(2*h**2)*np.sum(np.square(t1-t2),axis=1))

def laplacian_kernel(t1,t2,h=1):
    return np.exp(-1/(h)*np.sqrt(np.sum(np.square(t1-t2),axis=1)))

def matern_kernel(t1,t2,h=1):
    return (1+1/(h)*np.sqrt(np.sum(np.square(t1-t2),axis=1)))*np.exp(-1/(h)*np.sqrt(np.sum(np.square(t1-t2),axis=1)))

def sigmoid_func(x):
    return 1/(1+np.exp(-x))

for seed in range(200):
    path_output = "./output/partial_linear_rkhs_%d_%.3f_%d_%d.npz"%(n,sigma,mode,seed)
    path_train_data = "./data/partial_linear_train_%d_%.3f_%d_%d.npz"%(n,sigma,mode,seed)
    path_validation_data = "./data/partial_linear_validation_%d_%.3f_%d_%d.npz"%(n,sigma,mode,seed)
    path_test_data = "./data/partial_linear_test_%d_%.3f_%d_%d.npz"%(n,sigma,mode,seed)
    print(seed)
    trainset = model.Data(beta,round(0.8*n),sigma,mode=mode,func_f=func_f,seed = 20230915+seed)
    trainset.generate(path_train_data)
    print('Train data loaded')
    validationset = model.Data(beta,n-round(0.8*n),sigma,mode=mode,func_f=func_f,seed = 20230915+seed+10000)
    validationset.generate(path_validation_data)
    print('Validation data loaded')
    testset = model.Data(beta,n,sigma,mode=mode,func_f=func_f,seed = 20230915+seed+20000)
    testset.generate(path_test_data)
    print('test data loaded')
    
    if True:
        loss = 10000
        h = 0.2 * np.sqrt(d)
        for lamda in [1e-4,1e-3,1e-2,1e-1,1]:
            # print('==> training..')
            Phi = np.ndarray((trainset.len,trainset.len))
            for i in range(trainset.len):
                Phi[i,:] = laplacian_kernel(trainset.t_[i,:],trainset.t_,h).reshape(-1)
            X = trainset.x
            Y = trainset.data
            Mat = np.concatenate((X,Phi),axis = 1)
            I = np.zeros((trainset.len+2,trainset.len+2))
            for i in range(trainset.len):
                I[i+2,i+2] = 1
            Pen = lamda * I
            try:
                coef = np.zeros((trainset.len + 2,1))
                for iter in range(1000):
                    gradient = - np.matmul(np.transpose(Mat),(Y / (1e-3 + sigmoid_func(np.matmul(Mat,coef))))) + np.matmul(np.transpose(Mat),((1-Y) / (1e-3 + 1-sigmoid_func(np.matmul(Mat,coef)))))
                    coef[0:2,] = coef[0:2,] - 40*(gradient[0:2,]/trainset.len)/trainset.len
                    coef[2:,] = coef[2:,] - 40*(gradient[2:,]/trainset.len + lamda * coef[2:,])/trainset.len
                
                # coef = np.matmul(np.linalg.inv(np.matmul(np.transpose(Mat),Mat) + Pen),np.matmul(np.transpose(Mat),Y))
                beta_hat = coef[0:2,:]
                Y_fit = sigmoid_func(np.matmul(Mat,coef))
                train_loss = - np.mean(Y * np.log(1e-3 + Y_fit) + (1-Y) * np.log(1e-3 + 1-Y_fit))
                print('estimated beta:', beta_hat.reshape(-1))
                print('train loss:', train_loss)
                
                
                # print('==> validationing..')
                # validation
                Phi_validation = np.ndarray((validationset.len,trainset.len))
                for i in range(validationset.len):
                    Phi_validation[i,:] = laplacian_kernel(validationset.t_[i,:],trainset.t_,h).reshape(-1)
                X_validation =validationset.x
                Y_validation = validationset.data
                Mat_validation = np.concatenate((X_validation,Phi_validation),axis = 1)
                Y_fit = sigmoid_func(np.matmul(Mat_validation,coef))
                Y_validation = validationset.data
                validation_loss = - np.mean(Y_validation * np.log(1e-3 + Y_fit) + (1-Y_validation) * np.log(1e-3 + 1-Y_fit))
                print('validation loss:', validation_loss)
                
                if validation_loss < loss:
                    # print('==> saving..')
                    _coef = coef
                    # _h = h
                    _beta=beta_hat
                    loss = validation_loss
                    # _Y_fit = Y_fit
            except: 
                pass
            
        fit_Phi = np.ndarray((n,trainset.len))
        for i in range(n):
            fit_Phi[i,:] = laplacian_kernel(testset.t_[i,:],trainset.t_,h).reshape(-1)
        fit_f = np.matmul(fit_Phi,_coef[2:,:])
        para_value = _beta.reshape(-1)
        print(para_value)
        nonpara_value = np.mean(np.square(fit_f - testset.f))
        print(nonpara_value)
        np.savez(path_output,beta=para_value,non_loss=nonpara_value)
        