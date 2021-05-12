# CSBN
 Coupled Social and Biological Networks

### How do I get set up? ###
* Not sure if it works under MS Windows due to CuRAND library calls from device
* CUDA: https://developer.nvidia.com/cuda-downloads
* Python: https://www.python.org
* CuPy: https://docs.cupy.dev/en/stable/install.html 

### Files ###
* parameters.py parameter file
* csbn_run.py main file to run, distributes the work over several GPUs
  * csbn_gpus.py  calls network generation and epidemic runs in loops  
    * csbn_network.py => kernels_network.py, kernels_trn_network.py, kernels_gamma_network.py
    * csbn_network_barabasi.py => kernels_network_barabasi.py
    * csbn_epidemic.py => kernels_epidemic.py
* qstat.py  analyzis of dependence on q 
* network_check.py   network statistics (works only for N=100 or smaller)
* R0.py => kernels_R0.py, R0-params.csv Uses its own parameters instead of parameters.py
