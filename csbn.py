#
# for debuging: in terminal first enter: export CUDA_LAUNCH_BLOCKING=1
# Without it the error shows up in the next line, not where it occours
from csbn_network import *  # network creation part
from csbn_network_barabasi import *
from csbn_epidemic import *  # epidemic part
from parameters import *  # parameter file
from scipy import stats

import time
#starttime = time.time()
cp.cuda.Device(gpuNum).use()  # GPU used

blocksize_x = 1024 # maximum size of 1D block is 1024 threads

# The recovery process follows gamma distribution ip: max days it takes to recover, shape: average length 
disc=np.arange(ip+1); 
cumprob=stats.gamma.cdf(disc, shape, scale=1)
Pincubtrans=cp.asarray((cumprob[1:ip+1]-cumprob[0:ip])/(cumprob[ip]-cumprob[0:ip]), dtype=cp.float32)

# create data directory if it does not exist
import os
if not os.path.exists('data'):
    os.makedirs('data')

for netindx in range(NStart_netindx,NEnd_netindx+1):
    if network_run:
        if (network_type == "ban_cpmtx"):
            barabasifn (blocksize_x, N, Nsp_Children, Nsp_Parents, I0, Pc, Mc, Padd, Pret, netindx, network_save, 
                network_print)
        else:
            csbn_network(network_type, N, Nsp_Children, Nsp_Parents, Plink, Padd, Pret, I0, Pc, Mc, lambdaTheta,
                    blocksize_x, netindx, network_save, network_print)
        
        print("csbn_network from ",NStart_netindx," to ",NEnd_netindx, "Done: ",netindx)
    if epidemic_run:
        for q in np.arange(qmin,qmax+dq,dq,dtype=float):
            csbn_epidemic(N, I0, q, qeps, rho, Padv, aalpha, ggamma, bbeta, bbetah, NV0,
                  Peff, ssigma, gestation, MaxDays, ip, blocksize_x, netindx,
                  Pincubtrans, epidemic_save, epidemic_print)
        #print("csbn_epidemic from ", NStart_netindx," to ",NEnd_netindx, "Done: ",netindx)


#endtime = time.time()
#print("seconds/network: ", round((endtime - starttime)/(NEnd_netindx+1-NStart_netindx)))
