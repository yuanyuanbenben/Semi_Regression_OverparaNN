import numpy as np 
import copy
import os

if not os.path.isdir('data'):
    os.mkdir('data')
if not os.path.isdir('output'):
    os.mkdir('output')
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')

path_train_data = "./data/partial_linear_train.npz"
path_validation_data = "./data/partial_linear_validation.npz"
path_test_data = "./data/partial_linear_test.npz"

# Data            
raw_data_1 = np.genfromtxt('Aotizhongxin.csv', delimiter=',', skip_header=1)
raw_data_2 = np.genfromtxt('Changping.csv', delimiter=',', skip_header=1)
raw_data = np.concatenate((raw_data_1,raw_data_2),axis=0)

# split
x_data = copy.deepcopy(raw_data[:,(6,10)])
t_data = copy.deepcopy(raw_data[:,(4,5,7,8,9,12)])
y_data = copy.deepcopy(raw_data[:,3:4])

# normalization
x_data_mean = np.mean(x_data,axis=0)
x_data_std = np.std(x_data,axis=0)
x_data = (x_data - x_data_mean)/x_data_std
y_data_mean = np.mean(y_data,axis=0)
y_data_std = np.std(y_data,axis=0)
y_data = (y_data - y_data_mean)/y_data_std
# y_data = np.log(y_data)
t_data_mean = np.mean(t_data,axis=0)
t_data_std = np.std(t_data,axis=0)
t_data = (t_data - t_data_mean)/t_data_std

# random split to train,test,validation
np.random.seed(20250714)
n = 1435
index1 = np.random.choice(np.arange(1, 2869), size=n, replace=False)
index2 = np.ones(x_data.shape[0], dtype=bool)
index2[index1] = False
x_data_test = copy.deepcopy(x_data[index2,:])
y_data_test = copy.deepcopy(y_data[index2,:])
t_data_test = copy.deepcopy(t_data[index2,:])

# n1 = 0.8n
n1 = 1148
index3 = np.random.choice(np.arange(1, n), size=n1, replace=False)
index4 = np.ones(n, dtype=bool)
index5 = index1[index3]
index4[index3] = False
index6 = index1[index4]

x_data_train = copy.deepcopy(x_data[index5,:])
y_data_train = copy.deepcopy(y_data[index5,:])
t_data_train = copy.deepcopy(t_data[index5,:])

x_data_validation = copy.deepcopy(x_data[index6,:])
y_data_validation = copy.deepcopy(y_data[index6,:])
t_data_validation = copy.deepcopy(t_data[index6,:])

np.savez(path_train_data,x=x_data_train,y=y_data_train,t=t_data_train)
np.savez(path_validation_data,x=x_data_validation,y=y_data_validation,t=t_data_validation)
np.savez(path_test_data,x=x_data_test,y=y_data_test,t=t_data_test)