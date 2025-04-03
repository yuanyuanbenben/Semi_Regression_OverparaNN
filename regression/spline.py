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
parser.add_argument('--n','-n',default = 1000, type = int, help='sample size')
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

mode = args.mode
if mode == 0:
    d = 5
    func_f = lambda t_: ((t_[:,0]**2)*(t_[:,1]**3) + np.log(1+t_[:,2]) + np.sqrt(1+t_[:,3]*t_[:,4]) + np.exp(t_[:,4]/2))*5

if mode == 1:
    d = 10
    func_f = lambda t_: (((t_[:,0]**2)*(t_[:,1]**3) + np.log(1+t_[:,2]) + np.sqrt(1+t_[:,3]*t_[:,4]) + np.exp(t_[:,4]/2) + 
                         (t_[:,5]**2)*(t_[:,6]**3)/2 + np.log(1+t_[:,7])*2 + np.sqrt(1+t_[:,8]*t_[:,9])*2 + np.exp(t_[:,9]/2)/2))*5/2

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

for seed in range(200):
    path_output = "./output/partial_linear_spline_%d_%.3f_%d_%d.npz"%(n,sigma,mode,seed)
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
    
    if (mode == 0) or (mode == 8):
        loss = 100
        k_used = 0
        for k in range(2,10):
            if (k-1)**d > n:
                break
            # print('==> training..')
            # print('knots:',k)
            t = np.linspace(0,1,k,endpoint=True)
            t = np.r_[[0]*o,t,[1]*o]
            # Phi n*k
            _T = trainset.t_
            Phi1 = BSpline.design_matrix(_T[:,0].reshape(-1),t,o).toarray()
            Phi2 = BSpline.design_matrix(_T[:,1].reshape(-1),t,o).toarray()
            Phi3 = BSpline.design_matrix(_T[:,2].reshape(-1),t,o).toarray()
            Phi4 = BSpline.design_matrix(_T[:,3].reshape(-1),t,o).toarray()
            Phi5 = BSpline.design_matrix(_T[:,4].reshape(-1),t,o).toarray()
            
            def concat_phi(phi1,phi2,phi3,phi4,phi5):
                size1,size2 = phi1.shape
                ret = np.ndarray((size1,size2**5))
                for i in range(size1):
                    for j1 in range(size2):
                        for j2 in range(size2):
                            for j3 in range(size2):
                                for j4 in range(size2):
                                    for j5 in range(size2):
                                        ret[i,j1*size2**4+j2*size2**3+j3*size2**2+j4*size2+j5] = phi1[i,j1]*phi2[i,j2]*phi3[i,j3]*phi4[i,j4]*phi5[i,j5]
                return ret
                
            # X n*2
            X = trainset.x
            Y = trainset.data
            Mat = np.concatenate((X,concat_phi(Phi1,Phi2,Phi3,Phi4,Phi5)),axis = 1)
            try:
                coef = np.matmul(np.linalg.inv(np.matmul(np.transpose(Mat),Mat)),np.matmul(np.transpose(Mat),Y))
                beta_hat = coef[0:2,:]
                Y_fit = np.matmul(Mat,coef)
                train_loss = np.mean(np.square(Y-Y_fit))
                print('estimated beta:', beta_hat.reshape(-1))
                print('train loss:', train_loss)
                
                
                # print('==> validationing..')
                # validation
                _T_validation = validationset.t_
                Phi1_validation = BSpline.design_matrix(_T_validation[:,0].reshape(-1),t,o).toarray()
                Phi2_validation = BSpline.design_matrix(_T_validation[:,1].reshape(-1),t,o).toarray()
                Phi3_validation = BSpline.design_matrix(_T_validation[:,2].reshape(-1),t,o).toarray()
                Phi4_validation = BSpline.design_matrix(_T_validation[:,3].reshape(-1),t,o).toarray()
                Phi5_validation = BSpline.design_matrix(_T_validation[:,4].reshape(-1),t,o).toarray()
                X_validation = validationset.x
                Y_validation = validationset.data
                Mat_validation = np.concatenate((X_validation,concat_phi(Phi1_validation,Phi2_validation,Phi3_validation,Phi4_validation,Phi5_validation)),axis = 1)
                Y_fit = np.matmul(Mat_validation,coef)
                Y_validation = validationset.data
                validation_loss = np.mean(np.square(Y_validation-Y_fit))
                print('validation loss:', validation_loss)
                
                if validation_loss < loss:
                    # print('==> saving..')
                    _coef = coef
                    _t = t
                    _beta=beta_hat
                    _k=k
                    loss = validation_loss
                    _Y_fit = Y_fit
            except: 
                pass
        
        Phi1_test = BSpline.design_matrix(testset.t_[:,0].reshape(-1),_t,o).toarray()
        Phi2_test = BSpline.design_matrix(testset.t_[:,1].reshape(-1),_t,o).toarray()
        Phi3_test = BSpline.design_matrix(testset.t_[:,2].reshape(-1),_t,o).toarray()
        Phi4_test = BSpline.design_matrix(testset.t_[:,3].reshape(-1),_t,o).toarray()
        Phi5_test = BSpline.design_matrix(testset.t_[:,4].reshape(-1),_t,o).toarray()
        Phi = concat_phi(Phi1_test,Phi2_test,Phi3_test,Phi4_test,Phi5_test)
        fit_f = np.matmul(Phi,_coef[2:,:])
        para_value = _beta.reshape(-1)
        print(para_value)
        nonpara_value = np.mean(np.square(fit_f - testset.f))
        print(nonpara_value)
        np.savez(path_output,beta=para_value,non_loss=nonpara_value)

    if (mode == 1) or (mode == 9):
        loss = 100
        k_used = 0
        for k in range(2,10):
            if (k-1)**d > n:
                break
            # print('==> training..')
            # print('knots:',k)
            t = np.linspace(0,1,k,endpoint=True)
            t = np.r_[[0]*o,t,[1]*o]
            # Phi n*k
            _T = trainset.t_
            Phi1 = BSpline.design_matrix(_T[:,0].reshape(-1),t,o).toarray()
            Phi2 = BSpline.design_matrix(_T[:,1].reshape(-1),t,o).toarray()
            Phi3 = BSpline.design_matrix(_T[:,2].reshape(-1),t,o).toarray()
            Phi4 = BSpline.design_matrix(_T[:,3].reshape(-1),t,o).toarray()
            Phi5 = BSpline.design_matrix(_T[:,4].reshape(-1),t,o).toarray()
            Phi6 = BSpline.design_matrix(_T[:,5].reshape(-1),t,o).toarray()
            Phi7 = BSpline.design_matrix(_T[:,6].reshape(-1),t,o).toarray()
            Phi8 = BSpline.design_matrix(_T[:,7].reshape(-1),t,o).toarray()
            Phi9 = BSpline.design_matrix(_T[:,8].reshape(-1),t,o).toarray()
            Phi10 = BSpline.design_matrix(_T[:,9].reshape(-1),t,o).toarray()
            
            def concat_phi(phi1,phi2,phi3,phi4,phi5,phi6,phi7,phi8,phi9,phi10):
                size1,size2 = phi1.shape
                ret = np.ndarray((size1,size2**10))
                for i in range(size1):
                    print(i)
                    for j1 in range(size2):
                        for j2 in range(size2):
                            for j3 in range(size2):
                                for j4 in range(size2):
                                    for j5 in range(size2):
                                        for j6 in range(size2):
                                            for j7 in range(size2):
                                                for j8 in range(size2):
                                                    for j9 in range(size2):
                                                        for j10 in range(size2):
                                                            ret[i,j1*size2**9+j2*size2**8+j3*size2**7+j4*size2**6+j5*size2**5+j6*size2**4+j7*size2**3+j8*size2**2+j9*size2**1+j10] = phi1[i,j1]*phi2[i,j2]*phi3[i,j3]*phi4[i,j4]*phi5[i,j5]*phi1[i,j6]*phi2[i,j7]*phi3[i,j8]*phi4[i,j9]*phi5[i,j10]
                return ret
                
            # X n*2
            X = trainset.x
            Y = trainset.data
            Mat = np.concatenate((X,concat_phi(Phi1,Phi2,Phi3,Phi4,Phi5,Phi6, Phi7,Phi8,Phi9,Phi10)),axis = 1)
            print(Mat.shape)
            try:
                coef = np.matmul(np.linalg.inv(np.matmul(np.transpose(Mat),Mat)),np.matmul(np.transpose(Mat),Y))
                beta_hat = coef[0:2,:]
                Y_fit = np.matmul(Mat,coef)
                train_loss = np.mean(np.square(Y-Y_fit))
                print('estimated beta:', beta_hat.reshape(-1))
                print('train loss:', train_loss)
                
                
                # print('==> validationing..')
                # validation
                _T_validation = validationset.t_
                Phi1_validation = BSpline.design_matrix(_T_validation[:,0].reshape(-1),t,o).toarray()
                Phi2_validation = BSpline.design_matrix(_T_validation[:,1].reshape(-1),t,o).toarray()
                Phi3_validation = BSpline.design_matrix(_T_validation[:,2].reshape(-1),t,o).toarray()
                Phi4_validation = BSpline.design_matrix(_T_validation[:,3].reshape(-1),t,o).toarray()
                Phi5_validation = BSpline.design_matrix(_T_validation[:,4].reshape(-1),t,o).toarray()
                Phi6_validation = BSpline.design_matrix(_T_validation[:,5].reshape(-1),t,o).toarray()
                Phi7_validation = BSpline.design_matrix(_T_validation[:,6].reshape(-1),t,o).toarray()
                Phi8_validation = BSpline.design_matrix(_T_validation[:,7].reshape(-1),t,o).toarray()
                Phi9_validation = BSpline.design_matrix(_T_validation[:,8].reshape(-1),t,o).toarray()
                Phi10_validation = BSpline.design_matrix(_T_validation[:,9].reshape(-1),t,o).toarray()
                X_validation = validationset.x
                Y_validation = validationset.data
                Mat_validation = np.concatenate((X_validation,concat_phi(Phi1_validation,Phi2_validation,Phi3_validation,Phi4_validation,Phi5_validation,Phi6_validation,Phi7_validation,Phi8_validation,Phi9_validation,Phi10_validation)),axis = 1)
                Y_fit = np.matmul(Mat_validation,coef)
                Y_validation = validationset.data
                validation_loss = np.mean(np.square(Y_validation-Y_fit))
                print('validation loss:', validation_loss)
                
                if validation_loss < loss:
                    # print('==> saving..')
                    _coef = coef
                    _t = t
                    _beta=beta_hat
                    _k=k
                    loss = validation_loss
                    _Y_fit = Y_fit
            except: 
                pass
        
        Phi1_test = BSpline.design_matrix(testset.t_[:,0].reshape(-1),_t,o).toarray()
        Phi2_test = BSpline.design_matrix(testset.t_[:,1].reshape(-1),_t,o).toarray()
        Phi3_test = BSpline.design_matrix(testset.t_[:,2].reshape(-1),_t,o).toarray()
        Phi4_test = BSpline.design_matrix(testset.t_[:,3].reshape(-1),_t,o).toarray()
        Phi5_test = BSpline.design_matrix(testset.t_[:,4].reshape(-1),_t,o).toarray()
        Phi6_test = BSpline.design_matrix(testset.t_[:,5].reshape(-1),_t,o).toarray()
        Phi7_test = BSpline.design_matrix(testset.t_[:,6].reshape(-1),_t,o).toarray()
        Phi8_test = BSpline.design_matrix(testset.t_[:,7].reshape(-1),_t,o).toarray()
        Phi9_test = BSpline.design_matrix(testset.t_[:,8].reshape(-1),_t,o).toarray()
        Phi10_test = BSpline.design_matrix(testset.t_[:,9].reshape(-1),_t,o).toarray()
        Phi = concat_phi(Phi1_test,Phi2_test,Phi3_test,Phi4_test,Phi5_test,Phi6_test,Phi7_test,Phi8_test,Phi9_test,Phi10_test)
        fit_f = np.matmul(Phi,_coef[2:,:])
        para_value = _beta.reshape(-1)
        print(para_value)
        nonpara_value = np.mean(np.square(fit_f - testset.f))
        print(nonpara_value)
        np.savez(path_output,beta=para_value,non_loss=nonpara_value)