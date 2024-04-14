# CSBN
 Coupled Social and Biological Networks

Python CUDA code using the CuPy library to simulate parental decision-making about vaccinations against a childhood disease that spreads through a contact network. This model considers a bilayer network comprising two overlapping networks, which are either Erdős–Rényi (random) networks or Barabási–Albert networks. The model also employs a Bayesian aggregation rule for observational social learning on a social network. This new model encompasses other decision models, such as voting and DeGroot models, as special cases. Using our model, we demonstrate how certain levels of social learning about vaccination preferences can converge opinions, influencing vaccine uptake and ultimately disease spread. In addition, we explore how two different cultures of social learning affect the establishment of social norms of vaccination and the uptake of vaccines. In every scenario, the interplay between the dynamics of observational social learning and disease spread is influenced by the network’s topology, along with vaccine safety and availability.

## Publication:
Modeling the effect of observational social learning on parental decision-making for childhood vaccination and diseases spread over household networks
Tamer Oraby* and Andras Balogh
Front. Epidemiol., 12 January 2024
Sec. Infectious Disease Epidemiology
Volume 3 - 2023 | https://doi.org/10.3389/fepid.2023.1177752

### How do I get set up? 
* Not sure if it works under MS Windows due to CuRAND library calls from device
* CUDA: https://developer.nvidia.com/cuda-downloads
* Python: https://www.python.org
* CuPy: https://docs.cupy.dev/en/stable/install.html 

### Files ###
* parameters.py parameter file
* csbn_run.py main file to run, distributes the work over several GPUs
  * fcsbn.py  calls network generation and epidemic runs in loops  
    * csbn_network.py => kernels_network.py, kernels_trn_network.py, kernels_gamma_network.py
    * csbn_network_barabasi.py => kernels_network_barabasi.py
    * csbn_epidemic.py => kernels_epidemic.py
* qstat.py  analyzis of dependence on q 
* network_check.py   network statistics (works only for N=100 or smaller)
* R0_run.py => fR0.py, kernels_R0.py, R0-params.csv Uses its own parameters instead of parameters.py
### Running the code (short version)
* Edit the parameters.py file, set network_run=1, network_save=1, epidemic_run=0 
* Run csbn_run.py to generate only network
* Check  data/network_diagnostics....pdf
* Edit the parameters.py file, set set network_run=0, epidemic_run=1, epidemic_save=1 
* Run csbn_run.py to generate epidemics results
* Run qstat.py to generate analyzis of dependence on q values
* For calculating R0
  * Edit R0-params.csv
  * Edit the parameters in the file R0_run.py
  * Run the code R0_run.py

### Running the code (longer version)
* First networks are generated and saved in files data/csbn_network....npz, histogram "printed" in data/network_diagnostics....pdf 
* The work is distributed over several GPUs based on the number of network to be used.
* The epidemics part reads these netwrok files and runs the epidemics on them for a range of q values. results are saved in data/epidemics-q-...txt and "printed" in data/epidemics....pdf
* The network generation is much slower than the epidemics part
  * N=100,000: Network: 20 seconds/network, Epidemics: 2 seconds/network/q
  * N=1,000,000: Network: 20 minutes/network, Epidemics: 5 seconds/network/q
* Edit the parameters.py files then run csbn_run.py
  * For the first time network_run=1, network_save=1, network_print=1 are needed in order to generate and save the networks
* You might want to generate a single network by setting without running the epidemcs by
  * NStart_netindx=1 # starting index of network to generate
  * NEnd_netindx=1   # end index of network to generate
  * epidemics_run=0
  * Then view the network histogram in data/network_diagnostics001.pdf
  * If you are satisfied with the histogram then change  NEnd_netindx to larger value and rerun csbn_run.py
* Once the network files are generated edit parameters.py and set network_run=0, epidemic_run=1, epidemic_save=1, epidemic_print=1
* Run csbn_run.py again to generate files  data/epidemics-q-...txt and "printed" in data/epidemics....pdf
* Run qstat.py 
  * this reads the files  data/epidemics-q-...txt
  * writes q-....pdf files 
  
