#
# for debuging: in terminal first enter: export CUDA_LAUNCH_BLOCKING=1
# Without it the error shows up in the next line, not where it occours
from csbn_network import *  # network creation part
from csbn_network_barabasi import *
from csbn_epidemic import *  # epidemic part
from variables import *
from scipy import stats

import time
starttime = time.time()
cp.cuda.Device(gpuNum).use()  # GPU used

network_choice = network

# see csbn_cupy_notes.txt
N= population_size
Nsp_Children=20000000
Nsp_Parents=6000000 
#Plink=0.00028;   # children's network probability of a link 
#Padd=0.001;  # parents' network probability of adding a connection if children did not have one
#Pret=0.5    # parents' network probability of retaining children's connection
#I0=10 # Number of infected children

Plink=0.00013;   # children's network probability of a link 
Padd=0.0004;  # parents' network probability of adding a connection if children did not have one
Pret=0.6    # parents' network probability of retaining children's connection
I0=10 # Number of infected children

Pc=0.4 # 0.4 # Probability of having a child
Mc=7 # Maximum number of children in a family initially

blocksize_x = 1024 # maximum size of 1D block is 1024 threads
NStart_netindx=3 # starting index of network
NEnd_netindx=3   # end index of network

lambdaTheta = lt_trnExp # parameter for trn #-20

# epidemics parameters
q=0.5           # Probability of households' signal matching their vaccination opinion, pv_info_update
rho=0.01        # Probability of vaccination access, Vaccinate_Susceptibles
Padv=0.01       # Probability of adverse effect from vaccine, Vaccinate_Susceptibles
aalpha=10**(-4) # Household view of infection (encourages vaccination), p0
ggamma=0.1      # Household view of adverse effect (discourages vaccination), p0
bbeta=0.06      # Probability of transmission between households, Pinfection_update
bbetah=1.5*bbeta  # Probability of transmission within households, Pinfection_update
NV0=0.05        # Proportion of all-time never-vaccinator households, Vaccinator_yesnonever
Peff=0.45       # Probability of vaccine effectiveness (efficacy), Vaccinate_Susceptibles & Pregnancy_Newborns
ssigma=0.005    # Birth rate
gestation=280
MaxDays=1000
# The recovery process follows gamma distribution ip: days it takes to recover
#shape = 22; ip=28; disc=np.arange(ip+1); # Pertosis
shape = 11;  ip=16; disc=np.arange(ip+1); # Measles
cumprob=stats.gamma.cdf(disc, shape, scale=1)
Pincubtrans=cp.asarray((cumprob[1:ip+1]-cumprob[0:ip])/(cumprob[ip]-cumprob[0:ip]), dtype=cp.float32)

network_run=1; network_save=1; network_print=1
epidemic_run=1; epidemic_save=1; epidemic_print=1

# create data directory if it does not exist
import os
if not os.path.exists('data'):
    os.makedirs('data')


for netindx in range(NStart_netindx,NEnd_netindx+1):
    if network_run:
        if (network_choice == "ban_cpmtx"):
            barabasifn (blocksize_x, N, Nsp_Children, Nsp_Parents, I0, Pc, Mc, Padd, Pret, netindx, network_save, network_print)
        else:
            csbn_network(network_choice, N, Nsp_Children, Nsp_Parents, Plink, Padd, Pret, I0, Pc, Mc, lambdaTheta,
                    blocksize_x, netindx, network_save, network_print)
        
        print("csbn_network low: ",NStart_netindx," high: ",NEnd_netindx, "done: ",netindx)
    if epidemic_run:
        csbn_epidemic(N, I0, q, rho, Padv, aalpha, ggamma, bbeta, bbetah, NV0,
                  Peff, ssigma, gestation, MaxDays, ip, blocksize_x, netindx,
                  Pincubtrans, epidemic_save, epidemic_print)
        print("csbn_epidemic low: ", NStart_netindx," high: ",NEnd_netindx, "done: ",netindx)


endtime = time.time()
print("seconds/network: ", round((endtime - starttime)/(NEnd_netindx+1-NStart_netindx)))
