from csbn_network import *  # network creation part
from csbn_network_barabasi import *
from csbn_epidemic import *  # epidemic part
import cupy as cp
from parameters import *  # parameter file

def fcsbn(GPUnum, NStart_netindx, NEnd_netindx):
    try:
        print('Starting GPU {:1d} for networks {:2d}-{:2d}'.format(GPUnum, NStart_netindx, NEnd_netindx))
        cp.cuda.Device(GPUnum).use()  # GPU used
        for netindx in range(NStart_netindx, NEnd_netindx+1):
            if network_run:
                if (network_type == "ban_cpmtx"):
                    barabasifn(netindx)
                else:
                    csbn_network(network_type, netindx)

            if epidemic_run:
                if(delta!=9999):
                    q=0.0
                    csbn_epidemic(q, netindx)
                else:                 
                    for q in np.arange(qmin, qmax+dq, dq, dtype=float):
                        for qind in range(NQ):
                            csbn_epidemic(q, netindx)

        print('GPU {:1d} is finished for networks {:2d}-{:2d}'.format(GPUnum, NStart_netindx, NEnd_netindx))
    except:
        print("Something went wrong on GPU {:1d} for networks {:2d}-{:2d}".format(GPUnum, NStart_netindx, NEnd_netindx))
