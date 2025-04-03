# Code for "Semiparametric estimation with overparameterized neural networks"

#### ***Note***: This repository contain implementation code for the manuscript. The full source code is also publicly available on GitHub: [https://github.com/yuanyuanbenben/Semi_Regression_OverparaNN](https://github.com/yuanyuanbenben/Semi_Regression_OverparaNN). 

## Overview

- ### Directories ***regression*** and ***classification*** contains all codes in the simulation for the partial linear regression and classification examples, respectively. 
    - "***model.py***": Data generation processes and neural network architectures
    - "***main.py***": Core implementation of the proposed penalized M-estimation for semiparametric regression with overparameterized neural networks
    - "***spline.py***", "***kernel.py***", "***local_linear***" and "***nn.py***": The baselines 1,2,3 and 4 respectively.
    - "***test.ipynb***", "***results_out.py***": Results compilation and visualization in the manuscript. 
    - "***xxx.sh***": Shell files to run the same named python files. 

## Workflows
- Run the shell files directly (*./xxx.sh*) for our proposed method and baselines 1 to 4. For Cases 1 to 4, users need to change the *-m* parameter in shell files to 0,1,8,9 respectively. 
