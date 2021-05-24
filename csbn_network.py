# this rubroutine is called from csbn.py
# it creates the childrens and parents contact matrices 
import numpy as np
import cupy as cp
import os # needed for reading random seed form OS file /dev/random
import sys  # for sys.exit(0) 
from kernels_network import * # kernel functions (CUDA c++)
from kernels_trn_network import * 
from kernels_gamma_network import *
#from kernel_functions_epidemic import * # kernel functions (CUDA c++)
import math # for ceiling function

def csbn_network(network_func, N, Nsp_Children, Nsp_Parents, Plink, Padd, Pret, I0, Pc, Mc, lambdaTheta,
                 blocksize_x, netindx, network_save, network_print):
    blocks=(blocksize_x,1,1) 
    NT=(N*(N-1))//2 # Number of upper triangular entries
    NTchunk= 234881024 #335544320 234881024 117440512 16777216 
    NTiter=NT//NTchunk
    NTiterchunk=NTiter*NTchunk
    chunk_remainder=NT-NTiterchunk

    N_mtx_Children = 0 # will store the actual number of connections
    N_mtx_Parents = 0
    Susceptible=cp.random.binomial(Mc,Pc, size=N, dtype=cp.int32) 
    # distribute I0 infected randomly among the N households
    All_Infected=cp.zeros(N, dtype=cp.int32)
    Initial_Infected_Index=cp.random.choice(N, size=I0, replace=False,p=None) 
    All_Infected[Initial_Infected_Index]=1
    Children=All_Infected+Susceptible
    Children_mtx_indx=cp.zeros(Nsp_Children, dtype=cp.int64)
    Children_mtx_chunk=cp.zeros(NTchunk, dtype=cp.int32)
    Parents_mtx_indx=cp.zeros(Nsp_Parents, dtype=cp.int64)
    Parents_mtx_chunk=cp.zeros(NTchunk, dtype=cp.int32)
    grids=(math.ceil(NTchunk/blocksize_x),1,1) 
    network_types = {"ern_cpmtx":ern_cpmtx, "trnGamma_cpmtx":trnGamma_cpmtx, "trnExp_cpmtx":trnExp_cpmtx} 
    
    for i in range(NTiter):  # going through the chunks i=0,1,...,NTiter-1
        #print("Iteration",i," out of",NTiter)
        Children_mtx_chunk.fill(0.0) 
        Parents_mtx_chunk.fill(0.0) 
        NTshift=i*NTchunk # current starting index of the kernel 
        seed=int.from_bytes(os.urandom(4), 'big')  # get (new) random seed

        network_types[network_func](grids, blocks, (NTchunk, NTshift, cp.float32(Plink),
                    cp.float32(Pret), cp.float32(Padd), cp.float32(lambdaTheta), seed,
                    Children, Children_mtx_chunk, Parents_mtx_chunk))

        nzc=np.asscalar(cp.count_nonzero(Children_mtx_chunk).get()) # count how many nonzeros
        if (nzc>0):  # Children's connection matrix
            N_mtx_Children=N_mtx_Children+nzc
            if (N_mtx_Children>Nsp_Children):
                print("Error, must increase Nsp_Children!")
                sys.exit()
            cp.put(Children_mtx_indx, cp.arange(N_mtx_Children-nzc,
                                                stop=N_mtx_Children, step=1,
                                                dtype=cp.int64),
                   cp.add(cp.flatnonzero(Children_mtx_chunk), NTshift))
        nzp=np.asscalar(cp.count_nonzero(Parents_mtx_chunk).get())
        if (nzp>0):   # Parents' connection matrix
            N_mtx_Parents=N_mtx_Parents + nzp
            if (N_mtx_Parents>Nsp_Parents):
                print("Error, must increase Nsp_Parents!")
                sys.exit()
            cp.put(Parents_mtx_indx, cp.arange(N_mtx_Parents-nzp,
                                            stop=N_mtx_Parents, step=1,
                                            dtype=cp.int64),
                cp.add(cp.flatnonzero(Parents_mtx_chunk),NTshift))

    # remainder of the matrix
    if (chunk_remainder>0):
        grids=(math.ceil(chunk_remainder/blocksize_x),1,1) 
        Children_mtx_chunk.fill(0.0) 
        Parents_mtx_chunk.fill(0.0) 
        NTshift=NTiterchunk 
        seed=int.from_bytes(os.urandom(4), 'big')  # get (new) random seed
        
        network_types[network_func](grids, blocks, (chunk_remainder, NTshift, cp.float32(Plink),
                    cp.float32(Pret), cp.float32(Padd), cp.float32(lambdaTheta), seed,
                    Children, Children_mtx_chunk, Parents_mtx_chunk))

        nzc=np.asscalar(cp.count_nonzero(Children_mtx_chunk).get()) 

        if (nzc>0):
            N_mtx_Children=N_mtx_Children+nzc
            if (N_mtx_Children>Nsp_Children):
                print("Error, must increase Nsp_Children")
                sys.exit()
            cp.put(Children_mtx_indx, cp.arange(N_mtx_Children-nzc,
                                                stop=N_mtx_Children, step=1,
                                                dtype=cp.int64),
                   cp.add(cp.flatnonzero(Children_mtx_chunk),NTshift))
        nzp=np.asscalar(cp.count_nonzero(Parents_mtx_chunk).get())
        if (nzp>0):
            N_mtx_Parents=N_mtx_Parents+nzp 
            if (N_mtx_Parents>Nsp_Parents):
                print("Error, must increase Nsp_Parents")
                sys.exit()
            cp.put(Parents_mtx_indx, cp.arange(N_mtx_Parents-nzp,
                                            stop=N_mtx_Parents, step=1,
                                            dtype=cp.int64),
                cp.add(cp.flatnonzero(Parents_mtx_chunk),NTshift))
    if network_save:
        filename="data/csbn_network{:03d}".format(netindx)
        np.savez(filename, I0=I0, Infected=All_Infected.get(), N=N,
                Susceptible=Susceptible.get(), NC=N_mtx_Children,
                Children_mtx_indx=Children_mtx_indx[0:N_mtx_Children].get(),
                NP=N_mtx_Parents,
                Parents_mtx_indx=Parents_mtx_indx[0:N_mtx_Parents].get(),
                Plink=Plink, Pret=Pret, Padd=Padd)
    if network_print:
        from cupyx.scipy import sparse
        import matplotlib.pyplot as plt
        Jc = (cp.floor((1+cp.sqrt(8*Children_mtx_indx[0:N_mtx_Children]+1))/2)).astype(cp.int64)
        Ic = (Children_mtx_indx[0:N_mtx_Children] - ((Jc*(Jc-1))/2)).astype(cp.int64)
        # upper triangular part
        CP = sparse.coo_matrix((cp.ones(int(N_mtx_Children), dtype=cp.float32), (Ic, Jc)), 
                                shape=(N, N))
        csbn_network.CP = CP.tocsr()
        #print(csbn_network.CP)
        sC= cp.ravel(sparse.spmatrix.sum(CP, axis=0)+cp.transpose(sparse.spmatrix.sum(CP, axis=1)))
        
        Jp = (cp.floor((1+cp.sqrt(8*Parents_mtx_indx[0:N_mtx_Parents]+1))/2)).astype(cp.int64)
        Ip = (Parents_mtx_indx[0:N_mtx_Parents] - ((Jp*(Jp-1))/2)).astype(cp.int64)
        # upper triangular part
        PP = sparse.coo_matrix((cp.ones(int(N_mtx_Parents), dtype=cp.float32), (Ip, Jp)), 
                                shape=(N, N))
        sP= cp.ravel(sparse.spmatrix.sum(PP, axis=0)+cp.transpose(sparse.spmatrix.sum(PP, axis=1)))
        
        M=int(max(cp.amax(sC),np.amax(sP)))+1
        C_P_hist = plt.figure(1)
        plt.hist([sC.get(),sP.get()], bins=range(M+1), align='left',rwidth=0.9, label=["Children", "Parents"])
        plt.xlabel('Number of connections',fontsize=15)
        plt.ylabel('Frequency',fontsize=15)
        plt.grid(True)
        plt.legend(prop={'size': 10})
        plt.title("$P_{link}=$"+str(Plink)+", $P_{ret}=$"+str(Pret)+", $P_{add}=$"+str(Padd)+", $N=$"+str(N))

        filename="data/network_diagnostics{:03d}.pdf".format(netindx)
        plt.savefig(filename)
        plt.close()


