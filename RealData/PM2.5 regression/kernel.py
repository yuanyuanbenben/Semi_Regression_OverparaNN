import numpy as np
import copy
import os
# os.environ["OMP_NUM_THREADS"] = "32" # export OMP_NUM_THREADS=4
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

def gaussian_kernel(t1,t2,h=1):
    return np.exp(-1/(h**2)*np.sum(np.square(t1-t2),axis=1))

def laplacian_kernel(t1,t2,h=1):
    return np.exp(-1/(h)*np.sqrt(np.sum(np.square(t1-t2),axis=1)))

def matern_kernel(t1,t2,h=1):
    return (1+1/(h)*np.sqrt(np.sum(np.square(t1-t2),axis=1)))*np.exp(-1/(h)*np.sqrt(np.sum(np.square(t1-t2),axis=1)))

for seed in range(1):
    path_output = "./output/partial_linear_rkhs_%d.npz"%(n)
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
        h = 0.2*np.sqrt(d)
        for lamda in [1e-5,1e-4,1e-3,1e-2,1e-1,1]:
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
                coef = np.matmul(np.linalg.inv(np.matmul(np.transpose(Mat),Mat) + Pen),np.matmul(np.transpose(Mat),Y))
                beta_hat = coef[0:2,:]
                Y_fit = np.matmul(Mat,coef)
                train_loss = np.mean(np.square(Y-Y_fit))
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
                Y_fit = np.matmul(Mat_validation,coef)
                Y_validation = validationset.data
                validation_loss = np.mean(np.square(Y_validation-Y_fit))
                print('validation loss:', validation_loss)
                
                if validation_loss < loss:
                    # print('==> saving..')
                    _coef = coef
                    # _lamda = lamda
                    _beta=beta_hat
                    loss = validation_loss
                    _Y_fit = Y_fit
            except: 
                pass

        para_value = _beta.reshape(-1)
        print(para_value)
        Phi_test = np.ndarray((testset.len,trainset.len))
        for i in range(testset.len):
            Phi_test[i,:] = laplacian_kernel(testset.t_[i,:],trainset.t_,h).reshape(-1)
        X_test = testset.x
        Y_test = testset.data
        Mat_test = np.concatenate((X_test,Phi_test),axis = 1)
        Y_fit = np.matmul(Mat_test,_coef)
        Y_test = testset.data
        test_loss = np.mean(np.square(Y_test-Y_fit))
        print('test loss:', test_loss)
        np.savez(path_output,beta=para_value,test_loss=test_loss)
