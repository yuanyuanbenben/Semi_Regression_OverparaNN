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
parser.add_argument('--n','-n',default = 1435, type = int, help='sample size')
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

d = 6

print('==> Preparing data..')

def Epanechnikov_kernel(t1,t2,h,d):
    dis = np.mean(np.square(t1-t2),axis=1)/h
    index = np.ones_like(dis)
    index[dis>1] = 0
    return (1-dis)*index /((1-1/3/h)*2**d)

for seed in range(1):
    path_output = "./output/partial_linear_locallinear_%d.npz"%(n)
    path_train_data = "./data/partial_linear_train.npz"
    path_validation_data = "./data/partial_linear_validation.npz"
    path_test_data = "./data/partial_linear_test.npz"
    n_validation = 287
    n_test = 1434
    trainset = model.Data(beta,round(0.8*n),sigma)
    trainset.generate(path_train_data)
    print('Train data loaded')
    validationset = model.Data(beta,n_validation,sigma)
    validationset.generate(path_validation_data)
    print('Validation data loaded')
    testset = model.Data(beta,n_test,sigma)
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
        para_value = _beta.reshape(-1)
        print(para_value)

        Phi_test = np.ndarray((testset.len,trainset.len))
        for i in range(testset.len):
            Phi_test[i,:] = Epanechnikov_kernel(testset.t_[i,:],trainset.t_,h,d).reshape(-1)
        X_test = testset.x
        Y_test = testset.data
        S = np.ndarray((testset.len,trainset.len))
        for i in range(testset.len):
            Z0 = np.concatenate((np.ones((trainset.len,1)),trainset.t_ - testset.t_[i,:]),axis=1)
            Wh = np.diag(Phi_test[i,:])
            Z0_t_Wh = np.matmul(np.transpose(Z0),Wh)
            # pen = np.identity(d+1)
            S[i,:] = np.matmul(np.linalg.inv(np.matmul(Z0_t_Wh,Z0)),Z0_t_Wh)[0,:]
        Y_fit = np.matmul(S,(Y - np.matmul(X,beta_hat))) + np.matmul(X_test,beta_hat)
        test_loss = np.mean(np.square(Y_test-Y_fit))
        print('test loss:', test_loss)
        np.savez(path_output,beta=para_value,test_loss=test_loss)
