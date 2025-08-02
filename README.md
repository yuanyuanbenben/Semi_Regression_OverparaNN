# Materials for "Semiparametric estimation with overparameterized neural networks"
<a href="https://arxiv.org/pdf/2504.19089">https://arxiv.org/pdf/2504.19089</a>
## Overview

- ### Directories ***regression*** and ***classification*** contains all codes in the simulation for the partial linear regression and classification examples, respectively. 
    - "***model.py***": Data generation processes and neural network architectures
    - "***main.py***": Core implementation of the proposed penalized M-estimation for semiparametric regression with overparameterized neural networks
    - "***spline.py***", "***kernel.py***", "***local_linear***" and "***nn.py***": The baselines 1,2,3 and 4 respectively.
    - "***test.ipynb***", "***results_out.py***": Results compilation and visualization in the manuscript. 
    - "***xxx.sh***": Shell files to run the same named python files. 
- ### Directory  ***RealData***  contains codes and data for experiments on the real-world Beijing PM2.5 dataset.  
    - "***PM2.5 data***": Includes two monitoring site datasets, **Aotizhongxin** and **Changping**, publicly available at [Beijing Multi-Site Air-Quality Data (UCI)](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data).  
    - "***PM2.5 regression***" and "***PM2.5 classification***": 
        - "***model.py***": Data preprocessing and neural network architectures;
        - "***main.py***": Core implementation of the proposed penalized M-estimation for semiparametric regression with overparameterized neural networks;  
        - "***spline.py***", "***kernel.py***", "***local_linear.py***" and "***nn.py***": The code of baselines;
        - "***run.sh***": Shell files to run all python files;
        - "***plot.ipynb***": Results visualization in the manuscript.
## Workflows
- **Simulation:**
    - Run the shell files directly (`./xxx.sh`) for our proposed method and baselines 1 to 4.  
    - For Cases 1 to 4, users need to change the `-m` parameter in shell files to 0,1,8,9 respectively.  
- **RealData:**
    - Run `./run.sh` in either  PM2.5 regression  or  PM2.5 classification  folders to reproduce results.
    - Run `plot.ipynb` to visualize experimental results.
