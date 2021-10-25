from csbn_network import *  # network creation part
from csbn_network_barabasi import *
from csbn_epidemic import *  # epidemic part
import os
import sys
from scipy import stats
import cupy as cp
from parameters import *  # parameter file

def fcsbn(GPUnum, NStart_netindx, NEnd_netindx):
    try:
        print('Starting GPU {:1d} for networks {:2d}-{:2d}'.format(GPUnum, NStart_netindx, NEnd_netindx))
        cp.cuda.Device(GPUnum).use()  # GPU used
        # The recovery process follows gamma distribution ip: max days it takes to recover, mu: average length
        disc = np.arange(ip+1)
        cumprob = stats.gamma.cdf(disc, mu, scale=1)
        Pincubtrans = cp.asarray(
            (cumprob[1:ip+1]-cumprob[0:ip])/(cumprob[ip]-cumprob[0:ip]), dtype=cp.float32)
        for netindx in range(NStart_netindx, NEnd_netindx+1):
            if network_run:
                if (network_type == "ban_cpmtx"):
                    barabasifn(blocksize_x, N, Nsp_Children, Nsp_Parents, I0, Pc, Mc, Padd, Pret, netindx, network_save,
                       network_print)
                else:
                    csbn_network(network_type, N, Nsp_Children, Nsp_Parents, Plink, Padd, Pret, I0, Pc, Mc, lambdaTheta,
                         blocksize_x, netindx, network_save, network_print)

            if epidemic_run:
                for q in np.arange(qmin, qmax+dq, dq, dtype=float):
                    for qind in range(NQ):
                        csbn_epidemic(N, I0, q, qeps, rho, Padv, aalpha, ggamma, bbeta, bbetah, NV0,
                              Peff, ssigma, gestation, MaxDays, ip, blocksize_x, netindx,
                                Pincubtrans, delta, epidemic_save, epidemic_print)

        print('GPU {:1d} is finished for networks {:2d}-{:2d}'.format(GPUnum, NStart_netindx, NEnd_netindx))
    except:
        print("Something went wrong on GPU {:1d} for networks {:2d}-{:2d}".format(GPUnum, NStart_netindx, NEnd_netindx))
