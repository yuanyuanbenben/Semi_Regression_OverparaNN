import os
# os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=4

import numpy as np
import copy
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import argparse
# from scipy.interpolate import BSpline

import model

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--n','-n',default = 1000, type = int, help='sample size')
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
    func_f = lambda t_: ((t_[:,0]**2)*(t_[:,1]**3) + np.log(1+t_[:,2]) + np.sqrt(1+t_[:,3]*t_[:,4]) + np.exp(t_[:,4]/2))*5

if mode == 1:
    d = 10
    func_f = lambda t_: (((t_[:,0]**2)*(t_[:,1]**3) + np.log(1+t_[:,2]) + np.sqrt(1+t_[:,3]*t_[:,4]) + np.exp(t_[:,4]/2) + 
                        (t_[:,5]**2)*(t_[:,6]**3)/2 + np.log(1+t_[:,7])*2 + np.sqrt(1+t_[:,8]*t_[:,9])*2 + np.exp(t_[:,9]/2)/2))*5/2

if mode == 2:
    d = 5
    def func_f(t_):
        ret = np.zeros_like(t_[:,0])
        for i in range(30):
            for j in range(5):
                ret = ret + (i+1)**(-3) * np.cos((np.pi*2*t_[:,j]*(j*0.2+1)+j+1+np.pi*2*t_[:,(j+1)//d]*(j+1)*t_[:,(j+2)//d]*(j*0.2+1))*(i+1))
        return ret
    
if mode == 3:
    d = 10
    def func_f(t_):
        ret = np.zeros_like(t_[:,0])
        for i in range(30):
            for j in range(10):
                ret = ret + (i+1)**(-3) * np.cos((np.pi*2*t_[:,j]*(j*0.2+1)+j+1+np.pi*2*t_[:,(j+1)//d]*(j+1)*t_[:,(j+2)//d]*(j*0.2+1))*(i+1))
        return ret/2

if mode == 6:
    d = 5
    def func_f(t_):
        ret = np.zeros_like(t_[:,0])
        for i in range(30):
            for j in range(d):
                ret = ret + (i+1)**(-3) * np.cos((t_[:,j]*(j*0.1+1)+j+1)*(i+1)) * (j*0.1+0.75)
        return ret
    
if mode == 7:
    d = 10
    def func_f(t_):
        ret = np.zeros_like(t_[:,0])
        for i in range(30):
            for j in range(d):
                ret = ret + (i+1)**(-5.5) * np.cos((t_[:,j]*(j*0.1+1)+j+1)*(i+1)) * (j*0.1+0.5)
        return ret/2
    
if mode == 8:
    d = 5
    def func_f(t_):
        ret = np.zeros_like(t_[:,0])
        for j in range(d):
            ret = ret + (j+1)*t_[:,j]
        ret = np.sin(6*np.pi/d/(d+1)*ret)*5
        return ret
    
if mode == 9:
    d = 10
    def func_f(t_):
        ret = np.zeros_like(t_[:,0])
        for j in range(d):
            ret = ret + (j+1)*t_[:,j]
        ret = np.sin(6*np.pi/d/(d+1)*ret)*5
        return ret

print('==> Preparing data..')

def Epanechnikov_kernel(t1,t2,h,d):
    dis = np.mean(np.square(t1-t2),axis=1)/h
    index = np.ones_like(dis)
    index[dis>1] = 0
    return (1-dis)*index /((1-1/3/h)*2**d)

for seed in range(200):
    path_output = "./output/partial_linear_locallinear_%d_%.3f_%d_%d.npz"%(n,sigma,mode,seed)
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
        loss = 100
        for h in [0.1,1,10]:
            # print('==> training..')
            Phi = np.ndarray((trainset.len,trainset.len))
            for i in range(trainset.len):
                Phi[i,:] = Epanechnikov_kernel(trainset.t_[i,:],trainset.t_,h,d).reshape(-1)
            X = trainset.x
            Y = trainset.data
            I = np.identity(trainset.len)
            try:
                S = np.ndarray((trainset.len,trainset.len))
                for i in range(trainset.len):
                    Z0 = np.concatenate((np.ones((trainset.len,1)),trainset.t_ - trainset.t_[i,:]),axis=1)
                    Wh = np.diag(Phi[i,:])
                    Z0_t_Wh = np.matmul(np.transpose(Z0),Wh)
                    pen = np.identity(d+1) * 0.01
                    S[i,:] = np.matmul(np.linalg.inv(np.matmul(Z0_t_Wh,Z0)+pen),Z0_t_Wh)[0,:]
                
                X_t_I_min_S = np.matmul(np.transpose(X),np.matmul(np.transpose(I - S),I - S))
                beta_hat = np.matmul(np.linalg.inv(np.matmul(X_t_I_min_S,X)),np.matmul(X_t_I_min_S,Y))
                Y_fit = np.matmul(S,(Y - np.matmul(X,beta_hat))) + np.matmul(X,beta_hat)
                train_loss = np.mean(np.square(Y-Y_fit))
                print('estimated beta:', beta_hat.reshape(-1))
                print('train loss:', train_loss)
                
                
                # print('==> validationing..')
                # validation
                Phi_validation = np.ndarray((validationset.len,trainset.len))
                for i in range(validationset.len):
                    Phi_validation[i,:] = Epanechnikov_kernel(validationset.t_[i,:],trainset.t_,h,d).reshape(-1)
                X_validation =validationset.x
                Y_validation = validationset.data
                S = np.ndarray((validationset.len,trainset.len))
                for i in range(validationset.len):
                    Z0 = np.concatenate((np.ones((trainset.len,1)),trainset.t_ - validationset.t_[i,:]),axis=1)
                    Wh = np.diag(Phi_validation[i,:])
                    Z0_t_Wh = np.matmul(np.transpose(Z0),Wh)
                    # pen = np.identity(d+1)
                    S[i,:] = np.matmul(np.linalg.inv(np.matmul(Z0_t_Wh,Z0)),Z0_t_Wh)[0,:]
                Y_fit = np.matmul(S,(Y - np.matmul(X,beta_hat))) + np.matmul(X_validation,beta_hat)
                validation_loss = np.mean(np.square(Y_validation-Y_fit))
                print('validation loss:', validation_loss)
                
                if validation_loss < loss:
                    # print('==> saving..')
                    # _coef = coef
                    # _lamda = lamda
                    _h = h
                    _beta=beta_hat
                    loss = validation_loss
                    _Y_fit = Y_fit
            except:
                pass
            
        fit_Phi = np.ndarray((n,trainset.len))
        for i in range(n):
            fit_Phi[i,:] = Epanechnikov_kernel(testset.t_[i,:],trainset.t_,_h,d).reshape(-1)
        S = np.ndarray((n,trainset.len))
        for i in range(n):
            Z0 = np.concatenate((np.ones((trainset.len,1)),trainset.t_ - testset.t_[i,:]),axis=1)
            Wh = np.diag(fit_Phi[i,:])
            Z0_t_Wh = np.matmul(np.transpose(Z0),Wh)
            S[i,:] = np.matmul(np.linalg.inv(np.matmul(Z0_t_Wh,Z0)),Z0_t_Wh)[0,:]
        fit_f = np.matmul(S,(Y - np.matmul(X,_beta)))
        para_value = _beta.reshape(-1)
        print(para_value)
        nonpara_value = np.mean(np.square(fit_f - testset.f))
        print(nonpara_value)
        np.savez(path_output,beta=para_value,non_loss=nonpara_value)
