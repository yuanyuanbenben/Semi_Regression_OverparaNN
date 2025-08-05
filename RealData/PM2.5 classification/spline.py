import os
# os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=4
import numpy as np
import copy
import argparse
from scipy.interpolate import BSpline

import model

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--n','-n',default = 1435, type = int, help='sample size')
parser.add_argument('--sigma','-sigma','-s',default = 0.2, type = float, help='variance of noise')
parser.add_argument('--order','-o',default = 2, type = int, help='order of splines')
parser.add_argument('--mode','-m',default = 0, type = int, help='mode')
# parser.add_argument('--knots','-k',default = 50, type = int, help='knots of splines')
args = parser.parse_args()

# parameters setting
n = args.n
sigma = args.sigma
o = args.order
# k = args.knots

beta = np.array([[1],[0.75]])


if not os.path.isdir('data'):
    os.mkdir('data')
if not os.path.isdir('output'):
    os.mkdir('output')


d = 6
print('==> Preparing data..')

def sigmoid_func(x):
    return 1/(1+np.exp(-x))

for seed in range(1):
    path_output = "./output/partial_linear_spline_%d.npz"%(n)
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
        loss = 10000
        k_used = 0
        for k in range(2,3):
            if (k-1)**d > n:
                break
            # print('==> training..')
            # print('knots:',k)
            t = np.linspace(-10,10,k,endpoint=True)
            t = np.r_[[-10]*o,t,[10]*o]
            # Phi n*k
            _T = trainset.t_
            Phi1 = BSpline.design_matrix(_T[:,0].reshape(-1),t,o).toarray()
            Phi2 = BSpline.design_matrix(_T[:,1].reshape(-1),t,o).toarray()
            Phi3 = BSpline.design_matrix(_T[:,2].reshape(-1),t,o).toarray()
            Phi4 = BSpline.design_matrix(_T[:,3].reshape(-1),t,o).toarray()
            Phi5 = BSpline.design_matrix(_T[:,4].reshape(-1),t,o).toarray()
            Phi6 = BSpline.design_matrix(_T[:,5].reshape(-1),t,o).toarray()
            
            def concat_phi(phi1,phi2,phi3,phi4,phi5,phi6):
                size1,size2 = phi1.shape
                ret = np.ndarray((size1,size2**6))
                for i in range(size1):
                    for j1 in range(size2):
                        for j2 in range(size2):
                            for j3 in range(size2):
                                for j4 in range(size2):
                                    for j5 in range(size2):
                                        for j6 in range(size2):
                                            ret[i,j1*size2**5+j2*size2**4+j3*size2**3+j4*size2**2+j5*size2 + j6] = phi1[i,j1]*phi2[i,j2]*phi3[i,j3]*phi4[i,j4]*phi5[i,j5]*phi6[i,j6]
                return ret
                
            # X n*2
            X = trainset.x
            Y = trainset.data
            Mat = np.concatenate((X,concat_phi(Phi1,Phi2,Phi3,Phi4,Phi5,Phi6)),axis = 1)
            try:
                coef = np.zeros((Mat.shape[1],1))
                for iter in range(1000):
                    gradient = - np.matmul(np.transpose(Mat),(Y / (1e-2 + sigmoid_func(np.matmul(Mat,coef))))) + np.matmul(np.transpose(Mat),((1-Y) / (1e-2 + 1-sigmoid_func(np.matmul(Mat,coef)))))
                    coef = coef - (gradient/trainset.len)/trainset.len
                beta_hat = coef[0:2,:]
                Y_fit = sigmoid_func(np.matmul(Mat,coef))
                train_loss = - np.mean(Y * np.log(1e-2 + Y_fit) + (1-Y) * np.log(1e-2 + 1-Y_fit))
                print('estimated beta:', beta_hat.reshape(-1))
                print('train loss:', train_loss)
                
                
                # print('==> validationing..')
                # validation
                # _T_validation = validationset.t_
                # Phi1_validation = BSpline.design_matrix(_T_validation[:,0].reshape(-1),t,o).toarray()
                # Phi2_validation = BSpline.design_matrix(_T_validation[:,1].reshape(-1),t,o).toarray()
                # Phi3_validation = BSpline.design_matrix(_T_validation[:,2].reshape(-1),t,o).toarray()
                # Phi4_validation = BSpline.design_matrix(_T_validation[:,3].reshape(-1),t,o).toarray()
                # Phi5_validation = BSpline.design_matrix(_T_validation[:,4].reshape(-1),t,o).toarray()
                # X_validation = validationset.x
                # Y_validation = validationset.data
                # Mat_validation = np.concatenate((X_validation,concat_phi(Phi1_validation,Phi2_validation,Phi3_validation,Phi4_validation,Phi5_validation)),axis = 1)
                # Y_fit = np.matmul(Mat_validation,coef)
                # Y_validation = validationset.data
                # validation_loss = - np.mean(Y_validation * np.log(1e-2 + Y_fit) + (1-Y_validation) * np.log(1e-2 + 1-Y_fit))
                # print('validation loss:', validation_loss)
                
                # if validation_loss < loss:
                    # print('==> saving..')
                _coef = coef
                _t = t
                _beta=beta_hat
                _k=k
                # loss = validation_loss
                    # _Y_fit = Y_fit
            except: 
                pass
        
        para_value = _beta.reshape(-1)
        print(para_value)
        Phi1_test = BSpline.design_matrix(testset.t_[:,0].reshape(-1),_t,o).toarray()
        Phi2_test = BSpline.design_matrix(testset.t_[:,1].reshape(-1),_t,o).toarray()
        Phi3_test = BSpline.design_matrix(testset.t_[:,2].reshape(-1),_t,o).toarray()
        Phi4_test = BSpline.design_matrix(testset.t_[:,3].reshape(-1),_t,o).toarray()
        Phi5_test = BSpline.design_matrix(testset.t_[:,4].reshape(-1),_t,o).toarray()
        Phi6_test = BSpline.design_matrix(testset.t_[:,5].reshape(-1),_t,o).toarray()
        Phi = concat_phi(Phi1_test,Phi2_test,Phi3_test,Phi4_test,Phi5_test,Phi6_test)
        fit_f = np.matmul(Phi,_coef[2:,:]).reshape(-1)

        X_test = testset.x
        Y_test = testset.data
        Mat_test = np.concatenate((X_test,concat_phi(Phi1_test,Phi2_test,Phi3_test,Phi4_test,Phi5_test,Phi6_test)),axis = 1)
        Y_fit = sigmoid_func(np.matmul(Mat_test,coef))
        test_loss = - np.mean(Y_test * np.log(1e-2 + Y_fit) + (1-Y_test) * np.log(1e-2 + 1-Y_fit))
        print(test_loss)
        np.savez(path_output,beta=para_value,test_loss=test_loss)
        
        
        