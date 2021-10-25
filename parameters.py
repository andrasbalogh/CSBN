# This is the "parameters" file. 

gpuNum=8   # number of GPUs to distribute the work over. numbered as 0, 1, ..., gpuNum-1

# Comment out the desired network
network_type = "ern_cpmtx" 
#network_type = "ban_cpmtx"           
#network_type = "trnGamma_cpmtx" 
#network_type = "trnExp_cpmtx"

N = 100000 # population size
# I the sparse storage is not large enough the network generation stops with error message
# increasing it dynamically would slow down calculations
Nsp_Children=20000000 # Sparse storage size limit for Children's network. Adjust depending on N
Nsp_Parents=100000000 # Sparse storage size limit for Parents' network. Adjust depending on N, Plink, Padd

Plink=0.00013;   # children's network probability of a link 
Padd=0.0004;  # parents' network probability of adding a connection if children did not have one
Pret=0.6    # parents' network probability of retaining children's connection

# The code can generate several random networks saved in file data/csbn_network....npz
netindx_start=1 # starting index of network to generate
netindx_end=10   # end index of network to generate

I0=10 # Number of infected children

Pc=0.4 # Probability of having a child
Mc=7 # Maximum number of children in a family initially

lambdaTheta = -40.0 # parameter for treshold network #-20

# epidemics parameters
qmin = 0.1 # q - Probability of households' signal matching their vaccination opinion, pv_info_update 
qmax = 0.9 # endpoint included!!!
dq = 0.1
qeps=0.05        # (q-qeps, q+qeps) interval to generate random qij, qji 
# Repeat q's for social learning 
NQ=1
delta= 0.5      # Injunctive social pressure  (not used if delta =9999, in that case q is used)

rho=0.01        # Probability of vaccination access, Vaccinate_Susceptibles
Padv=0.0001       # Probability of adverse effect from vaccine, Vaccinate_Susceptibles
aalpha=10**(-4) # Household view of infection (encourages vaccination), p0
ggamma=0.1      # Household view of adverse effect (discourages vaccination), p0
bbeta=0.06      # Probability of transmission between households, Pinfection_update
bbetah=1.5*bbeta  # Probability of transmission within households, Pinfection_update
NV0=0.05        # Proportion of all-time never-vaccinator households, Vaccinator_yesnonever
Peff=0.45       # Probability of vaccine effectiveness (efficacy), Vaccinate_Susceptibles & Pregnancy_Newborns
ssigma=0.005    # Birth rate
gestation=280   # gestation time
MaxDays=1000    # storage size for daily epidemics statistics (code stops with error if it is not large enough)

# The recovery process follows gamma distribution ip: days it takes to recover, mu: average length
#mu = 22; ip=28; # Pertosis
mu = 11;  ip=16; # Measles

# note that the epidemic part always tries to read network that was previously saved into file! 
# once the files are generated there is no need to run network generation again
network_run=1 # run (1) or not run (0) the network generation
network_save=1 # save (1) or not save (0) generated network into file data/csbn_network....npz
network_print=1 # network histogram: save (1) or not save (0) into file data/network_diagnostics003.pdf

epidemic_run=0 # run (1) or not run (0) the epidemic on the network
epidemic_save=1 # saving epidemics data for using it in qstat.py
epidemic_print=1 # save plots of epidemics statistics into epidemcs....pdf

# Do not change it unless you know what you are doing
blocksize_x = 1024 
