import os
# os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import argparse
# from scipy.interpolate import BSpline
import numpy as np
import copy
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

def sigmoid_func(x):
    return 1/(1+np.exp(-x))

def temp_calculate(S,coef):
    return np.einsum('ijk,jks->is',S,coef)


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
        loss = 10000
        for h in [0.1,1,10]:
            # print('==> training..')
            Phi = np.ndarray((trainset.len,trainset.len))
            for i in range(trainset.len):
                Phi[i,:] = Epanechnikov_kernel(trainset.t_[i,:],trainset.t_,h,d).reshape(-1)
            X = trainset.x
            Y = trainset.data
            I = np.identity(trainset.len)
            try:
                S = np.ndarray((trainset.len,trainset.len,d+1))
                for i in range(trainset.len):
                    Z0 = np.concatenate((np.ones((trainset.len,1)),trainset.t_ - trainset.t_[i,:]),axis=1)
                    Wh = np.diag(Phi[i,:])
                    S[i,:,:] = np.transpose(np.matmul(np.transpose(Z0),Wh))
                
                coef_1 = np.zeros((2,1))
                coef_2 = np.zeros((trainset.len,d+1,1))
                for iter in range(1000):
                    part_1 = np.matmul(X,coef_1)
                    part_2 = temp_calculate(S,coef_2)
                    y_p = sigmoid_func(part_1 + part_2)
                    gradient_1 = - np.matmul(np.transpose(X),(Y / (1e-3 + y_p)))+ np.matmul(np.transpose(X),((1-Y) / (1e-3 + 1-y_p)))
                    gradient_2 = - np.einsum('jki,is->jks',np.transpose(S,(1,2,0)),(Y / (1e-3 + y_p))) + np.einsum('jki,is->jks',np.transpose(S,(1,2,0)),((1-Y) / (1e-3 + 1-y_p)))
                    coef_1 = coef_1 - 40*(gradient_1/trainset.len)/trainset.len
                    coef_2 = coef_2 - 40*(gradient_2/trainset.len)/trainset.len
                # coef = np.matmul(np.linalg.inv(np.matmul(np.transpose(Mat),Mat) + Pen),np.matmul(np.transpose(Mat),Y))
                beta_hat = coef_1
                Y_fit = sigmoid_func(np.matmul(X,beta_hat)+ temp_calculate(S,coef_2))
                train_loss = - np.mean(Y * np.log(1e-3 + Y_fit) + (1-Y) * np.log(1e-3 + 1-Y_fit))
                print('estimated beta:', beta_hat.reshape(-1))
                print('train loss:', train_loss)
                
                
                # print('==> validationing..')
                # validation
                Phi_validation = np.ndarray((validationset.len,trainset.len))
                for i in range(validationset.len):
                    Phi_validation[i,:] = Epanechnikov_kernel(validationset.t_[i,:],trainset.t_,h,d).reshape(-1)
                X_validation =validationset.x
                Y_validation = validationset.data
                S_validation = np.ndarray((validationset.len,trainset.len,d+1))
                for i in range(validationset.len):
                    Z0 = np.concatenate((np.ones((trainset.len,1)),trainset.t_ - validationset.t_[i,:]),axis=1)
                    Wh = np.diag(Phi_validation[i,:])
                    S_validation[i,:,:] = np.transpose(np.matmul(np.transpose(Z0),Wh))
                Y_fit = sigmoid_func(np.matmul(X_validation,beta_hat)+ temp_calculate(S_validation,coef_2))
                validation_loss = - np.mean(Y_validation * np.log(1e-3 + Y_fit) + (1-Y_validation) * np.log(1e-3 + 1-Y_fit))
                print('validation loss:', validation_loss)
                
                if validation_loss < loss:
                    # print('==> saving..')
                    # _coef = coef
                    # _lamda = lamda
                    _h = h
                    _beta=beta_hat
                    _coef = coef_2
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
        S_test = np.ndarray((testset.len,trainset.len,d+1))
        for i in range(testset.len):
            Z0 = np.concatenate((np.ones((trainset.len,1)),trainset.t_ - testset.t_[i,:]),axis=1)
            Wh = np.diag(Phi_test[i,:])
            S_test[i,:,:] = np.transpose(np.matmul(np.transpose(Z0),Wh))
        Y_fit = sigmoid_func(np.matmul(X_test,beta_hat)+ temp_calculate(S_test,coef_2))
        test_loss = - np.mean(Y_test * np.log(1e-3 + Y_fit) + (1-Y_test) * np.log(1e-3 + 1-Y_fit))
        print('test loss:', test_loss)
        np.savez(path_output,beta=para_value,test_loss=test_loss)
            
#     if (mode == 1) or (mode == 3):
#         loss = 100
#         for h in [1e-3,1e-2,1e-1,1,10]:
#             # print('==> training..')
#             Phi = np.ndarray((trainset.len,trainset.len))
#             for i in range(trainset.len):
#                 Phi[i,:] = Epanechnikov_kernel(trainset.t_[i,:],trainset.t_,h,d).reshape(-1)
#             X = trainset.x
#             Y = trainset.data
#             I = np.identity(trainset.len)
#             try:
#                 S = np.ndarray((trainset.len,trainset.len,d+1))
#                 for i in range(trainset.len):
#                     Z0 = np.concatenate((np.ones((trainset.len,1)),trainset.t_ - trainset.t_[i,:]),axis=1)
#                     Wh = np.diag(Phi[i,:])
#                     S[i,:,:] = np.transpose(np.matmul(np.transpose(Z0),Wh))
                
#                 coef_1 = np.zeros((2,1))
#                 coef_2 = np.zeros((trainset.len,d+1,1))
#                 for iter in range(10000):
#                     part_1 = np.matmul(X,coef_1)
#                     part_2 = temp_calculate(S,coef_2)
#                     y_p = sigmoid_func(part_1 + part_2)
#                     gradient_1 = - np.matmul(np.transpose(X),(Y / (1e-3 + y_p)))+ np.matmul(np.transpose(X),((1-Y) / (1e-3 + 1-y_p)))
#                     gradient_2 = - np.einsum('jki,is->jks',np.transpose(S,(1,2,0)),(Y / (1e-3 + y_p))) + np.einsum('jki,is->jks',np.transpose(S,(1,2,0)),((1-Y) / (1e-3 + 1-y_p)))
#                     coef_1 = coef_1 - 0.02*0.05/h*(gradient_1/trainset.len)
#                     coef_2 = coef_2 - 0.02*0.05/h*(gradient_2/trainset.len)
#                 # coef = np.matmul(np.linalg.inv(np.matmul(np.transpose(Mat),Mat) + Pen),np.matmul(np.transpose(Mat),Y))
#                 beta_hat = coef_1
#                 Y_fit = sigmoid_func(np.matmul(X,beta_hat)+ temp_calculate(S,coef_2))
#                 train_loss = - np.mean(Y * np.log(1e-3 + Y_fit) + (1-Y) * np.log(1e-3 + 1-Y_fit))
#                 print('estimated beta:', beta_hat.reshape(-1))
#                 print('train loss:', train_loss)
                
                
#                 # print('==> validationing..')
#                 # validation
#                 Phi_validation = np.ndarray((validationset.len,trainset.len))
#                 for i in range(validationset.len):
#                     Phi_validation[i,:] = Epanechnikov_kernel(validationset.t_[i,:],trainset.t_,h,d).reshape(-1)
#                 X_validation =validationset.x
#                 Y_validation = validationset.data
#                 S_validation = np.ndarray((validationset.len,trainset.len,d+1))
#                 for i in range(validationset.len):
#                     Z0 = np.concatenate((np.ones((trainset.len,1)),trainset.t_ - validationset.t_[i,:]),axis=1)
#                     Wh = np.diag(Phi_validation[i,:])
#                     S_validation[i,:,:] = np.transpose(np.matmul(np.transpose(Z0),Wh))
#                 Y_fit = sigmoid_func(np.matmul(X_validation,beta_hat)+ temp_calculate(S_validation,coef_2))
#                 validation_loss = np.mean(np.square(Y_validation-Y_fit))
#                 print('validation loss:', validation_loss)
                
#                 if validation_loss < loss:
#                     # print('==> saving..')
#                     # _coef = coef
#                     # _lamda = lamda
#                     _h = h
#                     _beta=beta_hat
#                     _coef = coef_2
#                     loss = validation_loss
#                     _Y_fit = Y_fit
#             except:
#                 pass
            
#         fit_Phi = np.ndarray((1001,trainset.len))
#         for i in range(1001):
#             fit_Phi[i,:] = Epanechnikov_kernel(test_t[i,:],trainset.t_,_h,d).reshape(-1)
#         S_fit = np.ndarray((1001,trainset.len,d+1))
#         for i in range(1001):
#             Z0 = np.concatenate((np.ones((trainset.len,1)),trainset.t_ - test_t[i,:]),axis=1)
#             Wh = np.diag(fit_Phi[i,:])
#             S_fit[i,:,:] = np.transpose(np.matmul(np.transpose(Z0),Wh))
#         fit_f = temp_calculate(S_fit,_coef)
#         para_value[seed,:] = _beta.reshape(-1)
#         print(para_value[seed,:])
#         nonpara_value[seed,:] = fit_f[:,0]
        
#         X_centered = X_validation - np.mean(X_validation)
#         Sigma_hat = np.mean(np.einsum('ij,isl->isl',np.square(Y_validation - _Y_fit),np.einsum('ij,ik->ijk',X_centered,X_centered)),axis=0,keepdims=False)
#         if np.matmul(np.transpose(_beta - beta),np.linalg.solve(Sigma_hat,_beta-beta)) > 5.991:
#             size_control  = size_control + 1
            
            
#     if (mode == 2) or (mode == 4):
#         loss = 100
#         for h in [1e-3,1e-2,1e-1,1,10]:
#             # print('==> training..')
#             Phi = np.ndarray((trainset.len,trainset.len))
#             for i in range(trainset.len):
#                 Phi[i,:] = Epanechnikov_kernel(trainset.t_[i,:],trainset.t_,h,d).reshape(-1)
#             X = trainset.x
#             Y = trainset.data
#             I = np.identity(trainset.len)
#             try:
#                 S = np.ndarray((trainset.len,trainset.len,d+1))
#                 for i in range(trainset.len):
#                     Z0 = np.concatenate((np.ones((trainset.len,1)),trainset.t_ - trainset.t_[i,:]),axis=1)
#                     Wh = np.diag(Phi[i,:])
#                     S[i,:,:] = np.transpose(np.matmul(np.transpose(Z0),Wh))
                
#                 coef_1 = np.zeros((2,1))
#                 coef_2 = np.zeros((trainset.len,d+1,1))
#                 for iter in range(10000):
#                     part_1 = np.matmul(X,coef_1)
#                     part_2 = temp_calculate(S,coef_2)
#                     y_p = sigmoid_func(part_1 + part_2)
#                     gradient_1 = - np.matmul(np.transpose(X),(Y / (1e-3 + y_p)))+ np.matmul(np.transpose(X),((1-Y) / (1e-3 + 1-y_p)))
#                     gradient_2 = - np.einsum('jki,is->jks',np.transpose(S,(1,2,0)),(Y / (1e-3 + y_p))) + np.einsum('jki,is->jks',np.transpose(S,(1,2,0)),((1-Y) / (1e-3 + 1-y_p)))
#                     coef_1 = coef_1 - 0.02*0.05/h*(gradient_1/trainset.len)
#                     coef_2 = coef_2 - 0.02*0.05/h*(gradient_2/trainset.len)
#                 # coef = np.matmul(np.linalg.inv(np.matmul(np.transpose(Mat),Mat) + Pen),np.matmul(np.transpose(Mat),Y))
#                 beta_hat = coef_1
#                 Y_fit = sigmoid_func(np.matmul(X,beta_hat)+ temp_calculate(S,coef_2))
#                 train_loss = - np.mean(Y * np.log(1e-3 + Y_fit) + (1-Y) * np.log(1e-3 + 1-Y_fit))
#                 print('estimated beta:', beta_hat.reshape(-1))
#                 print('train loss:', train_loss)
                
                
#                 # print('==> validationing..')
#                 # validation
#                 Phi_validation = np.ndarray((validationset.len,trainset.len))
#                 for i in range(validationset.len):
#                     Phi_validation[i,:] = Epanechnikov_kernel(validationset.t_[i,:],trainset.t_,h,d).reshape(-1)
#                 X_validation =validationset.x
#                 Y_validation = validationset.data
#                 S_validation = np.ndarray((validationset.len,trainset.len,d+1))
#                 for i in range(validationset.len):
#                     Z0 = np.concatenate((np.ones((trainset.len,1)),trainset.t_ - validationset.t_[i,:]),axis=1)
#                     Wh = np.diag(Phi_validation[i,:])
#                     S_validation[i,:,:] = np.transpose(np.matmul(np.transpose(Z0),Wh))
#                 Y_fit = sigmoid_func(np.matmul(X_validation,beta_hat)+ temp_calculate(S_validation,coef_2))
#                 validation_loss = np.mean(np.square(Y_validation-Y_fit))
#                 print('validation loss:', validation_loss)
                
#                 if validation_loss < loss:
#                     # print('==> saving..')
#                     # _coef = coef
#                     # _lamda = lamda
#                     _h = h
#                     _beta=beta_hat
#                     _coef = coef_2
#                     loss = validation_loss
#                     _Y_fit = Y_fit
#             except:
#                 pass
            
#         fit_Phi = np.ndarray((1001,trainset.len))
#         for i in range(1001):
#             fit_Phi[i,:] = Epanechnikov_kernel(test_t[i,:],trainset.t_,_h,d).reshape(-1)
#         S_fit = np.ndarray((1001,trainset.len,d+1))
#         for i in range(1001):
#             Z0 = np.concatenate((np.ones((trainset.len,1)),trainset.t_ - test_t[i,:]),axis=1)
#             Wh = np.diag(fit_Phi[i,:])
#             S_fit[i,:,:] = np.transpose(np.matmul(np.transpose(Z0),Wh))
#         fit_f = temp_calculate(S_fit,_coef)
#         para_value[seed,:] = _beta.reshape(-1)
#         print(para_value[seed,:])
#         nonpara_value[seed,:] = fit_f[:,0]
        
#         X_centered = X_validation - np.mean(X_validation)
#         Sigma_hat = np.mean(np.einsum('ij,isl->isl',np.square(Y_validation - _Y_fit),np.einsum('ij,ik->ijk',X_centered,X_centered)),axis=0,keepdims=False)
#         if np.matmul(np.transpose(_beta - beta),np.linalg.solve(Sigma_hat,_beta-beta)) > 5.991:
#             size_control  = size_control + 1
        
    
# np.savez(path_outputs,para_value=para_value,nonpara_value=nonpara_value,true_f = true_f,size = size_control/200)
                
        
        
        