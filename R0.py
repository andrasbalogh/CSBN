# R0
# for debuging: in terminal first enter: export CUDA_LAUNCH_BLOCKING=1
# Without it the error shows up in the next line, not where it occours
import os
import sys  # for stopping code if needed
from csbn_network import *  # ERN network creation part
from csbn_network_barabasi import *  # BAN network creation part
from csbn_epidemic import *  # epidemic part
from kernels_R0 import *
from scipy import stats
from cupyx.scipy import sparse
from csv import reader
import cupy as cp
import numpy as np
from numpy import load
import math  # for ceiling function

# Specifies what gpu to use
cp.cuda.Device(0).use()

# see csbn_cupy_notes.txt
N = 100000
Nsp_Children = 20000000

R0_samplesize = 110 # number of households to choose for sampling

# choosing rows from R0-params.csv 
rowstart = 1
rowend = 1

# Comment out the desired network
network_type = "ern_cpmtx" 
#network_type = "ban_cpmtx"           
#network_type = "trnGamma_cpmtx" 
#network_type = "trnExp_cpmtx"

lambdaTheta = -40.0 # parameter for trn #-20
Pc = 0.5  # 0.4 # Probability of having a child
Mc = 7  # Maximum number of children in a family initially
r0_repeat = 1  # how many times to repeat the disease process for averaging for each household
ssigma = 0.005    # Birth rate
gestation = 280
MaxDays = 100
# The recovery process follows gamma distribution ip: days it takes to recover
# shape = 22; ip=28;  # Pertosis
shape = 11; ip = 16 # Measles

network_print = 1
epidemic_save = 0
epidemic_print = 0
blocksize_x = 1024  # maximum size of 1D block is 1024 threads


disc = np.arange(ip+1)  
cumprob = stats.gamma.cdf(disc, shape, scale=1)
Pincubtrans = cp.asarray(
    (cumprob[1:ip+1]-cumprob[0:ip])/(cumprob[ip]-cumprob[0:ip]), dtype=cp.float32)

# create data directory if it does not exist
if not os.path.exists('data'):
    os.makedirs('data')

Plink = np.zeros(rowend+1, dtype=cp.float32)
bbeta = np.zeros(rowend+1, dtype=cp.float32)
h = np.zeros(rowend+1, dtype=cp.float32)
R0 = np.zeros(rowend+1, dtype=cp.float32)
# Probability of infection based on number of infected children in the neighborhood and in the household
P_infection = cp.zeros(N, dtype=cp.float32)

# open csv file in read mode
with open('R0-params.csv', 'r') as inputfile:
    # pass the file object to reader() to get the reader object
    rows = reader(inputfile)
    for row in range(rowstart):
        next(rows)     # discard these rows
    for i in range(rowstart, rowend+1):
        arow = next(rows)   # keep these rows
        Plink[i] = float(arow[0])
        bbeta[i] = float(arow[1])
        h[i] = float(arow[2])


blocks = (blocksize_x, 1, 1)
NT = (N*(N-1))//2  # Number of upper triangular entries
NTchunk = 234881024  # 335544320 234881024 117440512 16777216
NTiter = NT//NTchunk
NTiterchunk = NTiter*NTchunk
chunk_remainder = NT-NTiterchunk

#Children_mtx_indx = cp.zeros(Nsp_Children, dtype=cp.int64)
#Children_mtx_chunk = cp.zeros(NTchunk, dtype=cp.int32)
#Children = cp.zeros(N, dtype=cp.int32)
#Susceptible = cp.zeros(N, dtype=cp.int32)
AllInfected = cp.zeros(N, dtype=cp.int32)
Infected = cp.zeros((N, ip), dtype=cp.int32)
Recovered = cp.zeros((N, ip), dtype=cp.int32)
Pregnancy = cp.zeros(N, dtype=cp.int32)
Pregnancy0 = cp.zeros(N, dtype=cp.int32)
# Number of infected children of the neighbors
InfNeighb = cp.zeros(N, dtype=cp.int32)
nneighbs = cp.zeros((N), dtype=cp.int32)

# These settings should not be modified for R0 caculations
Nsp_Parents=1  # no parent networks needed, but the network code creates (a small) one 
I0=0
Padd=0.0 
Pret=0.0
network_save = 1

for row in range(rowstart, rowend+1):
    # creating network
    #Children0 = cp.random.binomial(Mc, Pc, size=N, dtype=cp.int32)
    #grids = (math.ceil(NTchunk/blocksize_x), 1, 1)
    #Children_mtx_indx.fill(0.0)
    Plinkrow = Plink[row]
    betarow = bbeta[row]
    betahrow = bbeta[row]*h[row]

    # call for network creation results are saved into files data/csbn_network(row).npz
    # Childrens/Susceptible/Initial_Infected_Index are created here!
    if (network_type == "ban_cpmtx"):
        barabasifn(blocksize_x, N, Nsp_Children, Nsp_Parents, I0, Pc, Mc, Padd, Pret, row, network_save, network_print)
    else:
        csbn_network(network_type, N, Nsp_Children, Nsp_Parents, Plinkrow, Padd, Pret, I0, Pc, Mc, lambdaTheta,
                     blocksize_x, row, network_save, network_print)

    #####################
    data=np.load("data/csbn_network{:03d}.npz".format(row))
    N_mtx_Children=data['NC'].item()  # Number of children connections, length of Children_mtx_indx
    Children_mtx_indx=cp.asarray(data['Children_mtx_indx'],dtype=cp.int64) # Length=NC, int64 is required because the indexing of the upper triangular entries goes from 0 to (N-1)*(N-2)/2
    Children0=cp.asarray(data['Susceptible'],dtype=cp.int32)  # All children are susceptible    
    ################
    
    Jc = (cp.floor((1+cp.sqrt(8*Children_mtx_indx[0:N_mtx_Children]+1))/2)).astype(cp.int64)
    Ic = (Children_mtx_indx[0:N_mtx_Children] - ((Jc*(Jc-1))/2)).astype(cp.int64)
    # upper triangular part
    CSP = sparse.coo_matrix((cp.ones(int(N_mtx_Children), dtype=cp.float32), (Ic, Jc)), shape=(N, N)).tocsr()

    MaxNeighb = int(cp.amax(cp.ravel(sparse.spmatrix.sum(CSP, axis=0)
                                     + cp.transpose(sparse.spmatrix.sum(CSP, axis=1)))))
    #print("MaxNeighb=",MaxNeighb)
    nneighbs.fill(0)
    neighbs = cp.zeros((N, MaxNeighb), dtype=cp.int32)
    # sample of households for placing one infected in
    R0_Index_Sample = np.random.choice(N, size=R0_samplesize, replace=False, p=None)
    for i in R0_Index_Sample:
        neighbs_temp = cp.concatenate(
            (CSP.getcol(i).tocsc().indices, CSP.getrow(i).indices))
        nneighbs[i] = neighbs_temp.size
        neighbs[i, 0:nneighbs[i]] = neighbs_temp
        # the neighbors' neighbors
        for j in neighbs_temp.get():
            neighbs_temp = cp.concatenate(
                (CSP.getcol(j).tocsc().indices, CSP.getrow(j).indices))
            nneighbs[j] = neighbs_temp.size
            neighbs[j, 0:nneighbs[j]] = neighbs_temp

    Pregnancy0.fill(0)
    grids = (math.ceil(N/blocksize_x), 1, 1)
    seed = int.from_bytes(os.urandom(4),'big') # Random seed from OS
    # Uses the number of existing children and birth rate (ssigma) to create pregnancies at different stages in each household.
    # Pregnancy[i] = 0 - no pregnancy
    # Pregnancy[i] = j, 0 < j < gestation - jth day of pregnancy
    pregnancy_burn_in(grids, blocks, (N, cp.float32(ssigma),  seed, Children0, Pregnancy0, gestation))

    # for sample use cp.nonzero(Children0[R0_Index_Sample] * nneighbs[R0_Index_Sample])[0]
    # for sample try using R0_Index_Sample
    # add if statement to continue if positive (if there are neighbs and current house has children)
    for i_house in R0_Index_Sample:
        if(Children0[i_house]*nneighbs[i_house] == 0): # skip the house with no children or no neighbor
            continue
        for r0loop in range(r0_repeat):
            # initialization
            Children = cp.copy(Children0)
            Susceptible = cp.copy(Children0)
            Pregnancy = cp.copy(Pregnancy0)
            Recovered.fill(0)
            AllInfected.fill(0)
            AllInfected[i_house] = 1
            Infected.fill(0)
            Infected[i_house, 0] = 1
            Infected_Total = 0
            InfNeighb.fill(0)
            P_Infection = 0.0
            Susceptible[i_house] = Susceptible[i_house] - 1
            day = 0
            indexcase = 1

            while indexcase > 0:
                day = day + 1
                if (day >= MaxDays):
                    sys.exit("Error, must increase MaxDays!")
                grids = (math.ceil(N/blocksize_x), 1, 1)  # set grid size N
                seed = int.from_bytes(os.urandom(4),'big') # Generates random seed from OS
                Pregnancy_Newborns(grids, blocks, (N, Susceptible, seed, Pregnancy,
                                                   gestation, Children, cp.float32(ssigma)))

                seed = int.from_bytes(os.urandom(4),'big') # Generates random seed from OS
                Recover_Infected(grids, blocks, (N, Infected, Recovered, seed, Pincubtrans, AllInfected, ip))

                grids = (math.ceil(N_mtx_Children/blocksize_x),  1, 1)  # set grid size NT
                InfNeighb.fill(0)
                Infected_Neighbors(grids, blocks, (N_mtx_Children, Children_mtx_indx, AllInfected,
                                                   InfNeighb))

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
                #print(i_house, nneighbs[i_house], neighbs[i_house,0:nneighbs[i_house]])
                for nb in neighbs[i_house, 0:nneighbs[i_house]]:
                    #print(Infected[nb,0],  nneighbs[nb])
                    if (Infected[nb, 0] and nneighbs[nb]):
                        R0[row] = R0[row]+(Infected[nb, 0]*betarow
                                           / (betarow+1.0-cp.power(1.0-betarow, AllInfected[i_house])
                                              + cp.sum(1.0-cp.power(1.0-betarow, cp.take(AllInfected, neighbs[nb, 0:nneighbs[nb]]) /
                                                                    cp.take(Children, neighbs[nb, 0:nneighbs[nb]])))
                                              - 1 + cp.power(1.0-betarow, AllInfected[i_house]/Children[i_house])))

                if day < ip:
                    indexcase = Infected[i_house, day]  # should be moved up ?
                else:
                    indexcase = 0

    R0[row] = R0[row]/(r0_repeat*R0_samplesize)
    print("row:", row, "Plink:", Plink[row], "beta:", betarow, "R0:", R0[row])


