import cupy as cp
import numpy as np
from scipy import stats
import csv
from csbn_network import *  # ERN network creation part
from csbn_network_barabasi import *  # BAN network creation part
from csbn_epidemic import *  # epidemic part
from kernels_R0 import *
from cupyx.scipy import sparse
import math  # for ceiling function
import sys  # for stopping code if needed
import os


Nsp_Parents=1  # no parent networks needed, but the network code creates (a small) one 
I0=0  # no infection at network generation, index case is placed one-by-one randomly at a later time
Padd=0.0 
Pret=0.0
network_save = 1

blocksize_x = 1024  # maximum size of 1D block is 1024 threads
blocks = (blocksize_x, 1, 1)


def fR0(GPUnum, rowstart, rowend, params):
    try:
        print('Starting GPU {:1d} for rows {:2d}-{:2d}'.format(GPUnum, rowstart, rowend))
        Nrows=rowend-rowstart+1
        cp.cuda.Device(GPUnum).use()  # GPU used
        ip=params[0]; shape=params[1]; N=params[2]; Nsp_Children=params[3]; Pc=params[4]; Mc=params[5]
        network_print=params[6]; lambdaTheta=params[7]; network_type=params[8]; R0_samplesize=params[9]
        ssigma=params[10]; gestation=params[11]; R0_repeat=params[12]; MaxDays=params[13]

        Plink = np.zeros(Nrows, dtype=cp.float32)
        bbeta = np.zeros(Nrows, dtype=cp.float32)
        h = np.zeros(Nrows, dtype=cp.float32)
        R0 = np.zeros(Nrows, dtype=cp.float32)
        # Probability of infection based on number of infected children in the neighborhood and in the household
        P_infection = cp.zeros(N, dtype=cp.float32)
        # open csv file in read mode
        with open('R0-params.csv', 'r') as inputfile:
            # pass the file object to reader() to get the reader object
            rows = csv.reader(inputfile)
            for row in range(rowstart):
                next(rows)     # discard these rows
            for i in range(rowend-rowstart+1):
                arow = next(rows)   # keep these rows
                Plink[i] = float(arow[0])
                bbeta[i] = float(arow[1])
                h[i] = float(arow[2])

        NT = (N*(N-1))//2  # Number of upper triangular entries
        NTchunk = 234881024  # 335544320 234881024 117440512 16777216
        NTiter = NT//NTchunk
        NTiterchunk = NTiter*NTchunk
        chunk_remainder = NT-NTiterchunk

        disc = np.arange(ip+1)  
        cumprob = stats.gamma.cdf(disc, shape, scale=1)
        Pincubtrans = cp.asarray((cumprob[1:ip+1]-cumprob[0:ip])/(cumprob[ip]-cumprob[0:ip]), dtype=cp.float32)

        AllInfected = cp.zeros(N, dtype=cp.int32)
        Infected = cp.zeros((N, ip), dtype=cp.int32)
        Recovered = cp.zeros((N, ip), dtype=cp.int32)
        Pregnancy0 = cp.zeros(N, dtype=cp.int32)
        # Number of infected children of the neighbors
        InfNeighb = cp.zeros(N, dtype=cp.int32)
        nneighbs = cp.zeros((N), dtype=cp.int32)

        for row in range(Nrows):
            # create network
            Plinkrow = Plink[row]
            betarow = bbeta[row]
            betahrow = bbeta[row]*h[row]
            # networks are saved into files data/csbn_network(rowstart+row).npz
            # Childrens/Susceptible/Initial_Infected_Index are created here!
            if (network_type == "ban_cpmtx"):
                barabasifn(blocksize_x, N, Nsp_Children, Nsp_Parents, I0, Pc, Mc, Padd, Pret, rowstart+row, network_save, 
                    network_print)
            else:
                csbn_network(network_type, N, Nsp_Children, Nsp_Parents, Plinkrow, Padd, Pret, I0, Pc, Mc, 
                    lambdaTheta, blocksize_x, rowstart+row, network_save, network_print)

            ######### load the network back from file ############
            data=np.load("data/csbn_network{:03d}.npz".format(rowstart+row))
            N_mtx_Children=data['NC'].item()  # Number of children connections, length of Children_mtx_indx
            Children_mtx_indx=cp.asarray(data['Children_mtx_indx'],dtype=cp.int64) 
            Children0=cp.asarray(data['Susceptible'],dtype=cp.int32)  # All children are susceptible    
            ################
            # row Ic and column Jc indices for sparse storage
            Jc = (cp.floor((1+cp.sqrt(8*Children_mtx_indx[0:N_mtx_Children]+1))/2)).astype(cp.int64)
            Ic = (Children_mtx_indx[0:N_mtx_Children] - ((Jc*(Jc-1))/2)).astype(cp.int64)
            # upper triangular part
            CSP = sparse.coo_matrix((cp.ones(int(N_mtx_Children), dtype=cp.float32), (Ic, Jc)), shape=(N, N)).tocsr()
            # for the whole matrix we have to add the lower triangular (transposed) part
            MaxNeighb = int(cp.amax(cp.ravel(sparse.spmatrix.sum(CSP, axis=0)
                                 + cp.transpose(sparse.spmatrix.sum(CSP, axis=1)))))
            nneighbs.fill(0)
            neighbs = cp.zeros((N, MaxNeighb), dtype=cp.int32)
            # sample of households for placing one infected in
            R0_Index_Sample = np.random.choice(N, size=R0_samplesize, replace=False, p=None)
            for i in R0_Index_Sample:
                neighbs_temp = cp.concatenate((CSP.getcol(i).tocsc().indices, CSP.getrow(i).indices))
                nneighbs[i] = neighbs_temp.size
                neighbs[i, 0:nneighbs[i]] = neighbs_temp
                # the neighbors' neighbors
                for j in neighbs_temp.get():
                    neighbs_temp = cp.concatenate((CSP.getcol(j).tocsc().indices, CSP.getrow(j).indices))
                    nneighbs[j] = neighbs_temp.size
                    neighbs[j, 0:nneighbs[j]] = neighbs_temp

            Pregnancy0.fill(0)
            grids = (math.ceil(N/blocksize_x), 1, 1)
            seed = int.from_bytes(os.urandom(4),'big') # Random seed from OS
            pregnancy_burn_in(grids, blocks, (N, cp.float32(ssigma),  seed, Children0, Pregnancy0, gestation))

            for i_house in R0_Index_Sample:
                if(Children0[i_house]*nneighbs[i_house] == 0): # skip the house with no children or no neighbor
                    continue
                for r0loop in range(R0_repeat):
                    # initialization
                    Children = cp.copy(Children0)
                    Susceptible = cp.copy(Children0) # all children are susceptable (-1 index case later) 
                    Pregnancy = cp.copy(Pregnancy0)
                    Recovered.fill(0)
                    AllInfected.fill(0) 
                    AllInfected[i_house] = 1  # place one infected into i_house household
                    Infected.fill(0)
                    Infected[i_house, 0] = 1
                    Infected_Total = 0
                    InfNeighb.fill(0)
                    P_infection.fill(0.0)
                    Susceptible[i_house] = Susceptible[i_house] - 1
                    day = 0
                    indexcase = 1

                    while indexcase > 0:
                        day = day + 1
                        if (day >= MaxDays):
                            print("Error, must increase MaxDays!")
                            sys.exit(0)

                        grids = (math.ceil(N/blocksize_x), 1, 1)  # set grid size N
                        seed = int.from_bytes(os.urandom(4),'big') # Generates random seed from OS
                        Pregnancy_Newborns(grids, blocks, (N, Susceptible, seed, Pregnancy, gestation, 
                                Children, cp.float32(ssigma)))

                        seed = int.from_bytes(os.urandom(4),'big') # Generates random seed from OS
                        Recover_Infected(grids, blocks, (N, Infected, Recovered, seed, Pincubtrans, AllInfected, ip))

                        grids = (math.ceil(N_mtx_Children/blocksize_x),  1, 1)  # set grid size NT
                        InfNeighb.fill(0)
                        Infected_Neighbors(grids, blocks, (N_mtx_Children, Children_mtx_indx, AllInfected, InfNeighb))

                        grids = (math.ceil(N/blocksize_x), 1, 1)  # set grid size NT
                        Pinfection_update(grids, blocks, (N, P_infection, InfNeighb, Children,
                                                  AllInfected, cp.float32(betarow), cp.float32(betahrow)))


                        seed = int.from_bytes(os.urandom(4),'big') # Generates random seed from OS
                        New_Infected(grids, blocks, (N, P_infection, Infected, Susceptible, seed, AllInfected, ip))

                        # i_house
                        R0[row] = R0[row]+(Infected[i_house, 0]*betahrow
                                   / (betahrow+1.0-cp.power(1.0-betahrow, AllInfected[i_house])
                                      + cp.sum(1.0-cp.power(1.0-betarow, cp.take(AllInfected, neighbs[i_house, 0:nneighbs[i_house]]) /
                                                            cp.take(Children, neighbs[i_house, 0:nneighbs[i_house]])))))

                        # the neighbors (take out i_house!!!!) double check formulas
                        for nb in neighbs[i_house, 0:nneighbs[i_house]]:
                            if (Infected[nb, 0] and nneighbs[nb]):
                                R0[row] = R0[row]+(Infected[nb, 0]*betarow
                                           / (betarow+1.0-cp.power(1.0-betarow, AllInfected[i_house])
                                              + cp.sum(1.0-cp.power(1.0-betarow, cp.take(AllInfected, neighbs[nb, 0:nneighbs[nb]]) /
                                                                    cp.take(Children, neighbs[nb, 0:nneighbs[nb]])))
                                              - 1 + cp.power(1.0-betarow, AllInfected[i_house]/Children[i_house])))

                        if day < ip:
                            indexcase = Infected[i_house, day] 
                        else:
                            indexcase = 0

            R0[row] = R0[row]/(R0_repeat*R0_samplesize)
            print("row:", rowstart+row, "N:", N, "Plink:", Plink[row], "beta:", betarow, "h:", h[row], "R0:", R0[row])
            with open('R0-results.csv', 'a', newline='') as csvfile: # appends to the file
                r0writer = csv.writer(csvfile, delimiter=' ')
                r0writer.writerow([rowstart+row, ",", N, ",", Plink[row], ",", betarow, ",", h[row], ",", R0[row]])

        print('GPU {:1d} is finished for rows {:2d}-{:2d}'.format(GPUnum, rowstart, rowend))
    except:
        print("Something went wrong on GPU {:1d} for rows {:2d}-{:2d}".format(GPUnum, rowstart, rowend))
