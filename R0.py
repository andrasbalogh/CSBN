# R0
# for debuging: in terminal first enter: export CUDA_LAUNCH_BLOCKING=1
# Without it the error shows up in the next line, not where it occours
# from csbn_R0 import *  # epidemic part
import os
import sys  # for stopping code if needed
from kernels_R0 import *
from scipy import stats
from cupyx.scipy import sparse
from csv import reader
import cupy as cp
import numpy as np
import math  # for ceiling function
import time
starttime = time.time()

# Specifies what gpu to use
cp.cuda.Device(7).use()

# see csbn_cupy_notes.txt
N = 1000
Nsp_Children = 2000000
Pc = 0.5  # 0.4 # Probability of having a child
Mc = 7  # Maximum number of children in a family initially
r0_repeat = 1  # how many times to repeat the disease process for averaging for each household
ssigma = 0.005    # Birth rate
gestation = 280
MaxDays = 100
# The recovery process follows gamma distribution ip: days it takes to recover
# shape = 22; ip=28; disc=np.arange(ip+1); # Pertosis
shape = 11
ip = 16
disc = np.arange(ip+1)  # Measles
cumprob = stats.gamma.cdf(disc, shape, scale=1)
Pincubtrans = cp.asarray(
    (cumprob[1:ip+1]-cumprob[0:ip])/(cumprob[ip]-cumprob[0:ip]), dtype=cp.float32)

network_save = 0
network_print = 0
epidemic_save = 0
epidemic_print = 0
blocksize_x = 1024  # maximum size of 1D block is 1024 threads

# create data directory if it does not exist
if not os.path.exists('data'):
    os.makedirs('data')

rowstart = 1
rowend = 1
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
        # print(Plink[i]*100)
        # print(cp.float32(Plink[i]),beta[i],h[i])


blocks = (blocksize_x, 1, 1)
NT = (N*(N-1))//2  # Number of upper triangular entries
NTchunk = 234881024  # 335544320 234881024 117440512 16777216
NTiter = NT//NTchunk
NTiterchunk = NTiter*NTchunk
chunk_remainder = NT-NTiterchunk

# sys.exit("Done!")
Children_mtx_indx = cp.zeros(Nsp_Children, dtype=cp.int64)
Children_mtx_chunk = cp.zeros(NTchunk, dtype=cp.int32)
Children = cp.zeros(N, dtype=cp.int32)
Susceptible = cp.zeros(N, dtype=cp.int32)
AllInfected = cp.zeros(N, dtype=cp.int32)
Infected = cp.zeros((N, ip), dtype=cp.int32)
Recovered = cp.zeros((N, ip), dtype=cp.int32)
Pregnancy = cp.zeros(N, dtype=cp.int32)
Pregnancy0 = cp.zeros(N, dtype=cp.int32)
# Number of infected children of the neighbors
InfNeighb = cp.zeros(N, dtype=cp.int32)
nneighbs = cp.zeros((N), dtype=cp.int32)


for row in range(rowstart, rowend+1):
    # creating network
    t4 = time.time()
    # print("row:",row,"Plink:",Plink[row],"beta:",bbeta[row],"h:", h[row])
    N_mtx_Children = 0  # will store the actual number of connections
    Children0 = cp.random.binomial(Mc, Pc, size=N, dtype=cp.int32)
    grids = (math.ceil(NTchunk/blocksize_x), 1, 1)
    Children_mtx_indx.fill(0.0)
    Plinkrow = Plink[row]
    betarow = bbeta[row]
    betahrow = bbeta[row]*h[row]
    for i in range(NTiter):  # going through the chunks i=0,1,...,NTiter-1
        # print("Iteration",i," out of",NTiter)
        Children_mtx_chunk.fill(0.0)
        NTshift = i*NTchunk  # current starting index of the kernel
        seed = 1
        # int.from_bytes(os.urandom(4), 'big')  # get (new) random seed
        setcpmtxR0(grids, blocks, (NTchunk, NTshift, cp.float32(Plinkrow),
                                   seed, Children0, Children_mtx_chunk))  # creates 0/1 arrays
        # count how many nonzeros
        nzc = cp.count_nonzero(Children_mtx_chunk).get().item()
        if (nzc > 0):  # Children's connection matrix
            N_mtx_Children = N_mtx_Children+nzc
            if (N_mtx_Children > Nsp_Children):
                sys.exit("Error, must increase Nsp_Children!")
            cp.put(Children_mtx_indx, cp.arange(N_mtx_Children-nzc,
                                                stop=N_mtx_Children, step=1,
                                                dtype=cp.int64),
                   cp.add(cp.flatnonzero(Children_mtx_chunk), NTshift))

    # remainder of the matrix
    if (chunk_remainder > 0):
        grids = (math.ceil(chunk_remainder/blocksize_x), 1, 1)
        Children_mtx_chunk.fill(0.0)
        NTshift = NTiterchunk
        seed = 1
        # int.from_bytes(os.urandom(4), 'big')  # get (new) random seed
        setcpmtxR0(grids, blocks, (chunk_remainder, NTshift, cp.float32(Plinkrow),
                                   seed, Children0, Children_mtx_chunk))
        nzc = cp.count_nonzero(Children_mtx_chunk).get().item()
        if (nzc > 0):
            N_mtx_Children = N_mtx_Children+nzc
            if (N_mtx_Children > Nsp_Children):
                sys.exit("Error, must increase Nsp_Children")
            cp.put(Children_mtx_indx, cp.arange(N_mtx_Children-nzc,
                                                stop=N_mtx_Children, step=1,
                                                dtype=cp.int64),
                   cp.add(cp.flatnonzero(Children_mtx_chunk), NTshift))

    Jc = (cp.floor(
        (1+cp.sqrt(8*Children_mtx_indx[0:N_mtx_Children]+1))/2)).astype(cp.int64)
    Ic = (Children_mtx_indx[0:N_mtx_Children] -
          ((Jc*(Jc-1))/2)).astype(cp.int64)
    # upper triangular part in coordinate format
    CSP = sparse.coo_matrix((cp.ones(int(N_mtx_Children), dtype=cp.float32), (Ic, Jc)),
                            shape=(N, N))
    CSP = CSP.tocsr()  # convert to compresses sparse row format
    if network_save:
        filename = "data/csbn_network{:03d}".format(row)
        np.savez(filename, N=N, Children=Children0.get(), NC=N_mtx_Children,
                 Children_mtx_indx=Children_mtx_indx[0:N_mtx_Children].get(),
                 Plink=Plinkrow)
    if network_print:
        import matplotlib.pyplot as plt
        # add rows and columns of the upper triangular part
        # to get total number of connections for each household
        sC = cp.ravel(sparse.spmatrix.sum(CSP, axis=0) +
                      cp.transpose(sparse.spmatrix.sum(CSP, axis=1)))
        C_P_hist = plt.figure(1)
        plt.hist([sC.get()], bins=range(int(cp.amax(sC))+2),
                 align='left', rwidth=0.9, label=["Children"])
        plt.xlabel('Number of connections', fontsize=15)
        plt.ylabel('Frequency', fontsize=15)
        plt.grid(True)
        plt.legend(prop={'size': 10})
        plt.title("$P_{link}=$"+str(Plinkrow)+", $N=$"+str(N))
        filename = "data/network_diagnostics{:03d}.pdf".format(row)
        plt.savefig(filename)
        plt.close()
        # below code visualizes network
        # plt.figure(2)
        #import networkx as nx
        #njc=Jc.get(); nic=Ic.get()
        #G = nx.Graph()
        # for i in range(N_mtx_Children):
        # print(nip[i], njp[i])
        #    G.add_edge(nic[i], njc[i])
        #nx.draw(G, with_labels=True)
        # plt.show()

    MaxNeighb = int(cp.amax(cp.ravel(sparse.spmatrix.sum(CSP, axis=0)
                                     + cp.transpose(sparse.spmatrix.sum(CSP, axis=1)))))
    nneighbs.fill(0)
    neighbs = cp.zeros((N, MaxNeighb), dtype=cp.int32)
    for i in range(N):
        neighbs1 = cp.concatenate(
            (CSP.getcol(i).tocsc().indices, CSP.getrow(i).indices))
        nneighbs[i] = neighbs1.size
        neighbs[i, 0:nneighbs[i]] = neighbs1
    Pregnancy0.fill(0)
    grids = (math.ceil(N/blocksize_x), 1, 1)
    seed = 1
    # int.from_bytes(os.urandom(4),'big') # Random seed from OS
    # Uses the number of existing children and birth rate (ssigma) to create pregnancies at different stages in each household.
    # Pregnancy[i] = 0 - no pregnancy
    # Pregnancy[i] = j, 0 < j < gestation - jth day of pregnancy
    pregnancy_burn_in(grids, blocks, (N, cp.float32(ssigma),
                                      seed, Children0, Pregnancy0, gestation))
    t0 = time.time()
    print("Time to create network: ", (t0-t4))
    N_Index = 100
    R0_Index_Sample=cp.random.choice(N, size=N_Index, replace=False,p=None) 
    #for sample use cp.nonzero(Children0[R0_Index_Sample] * nneighbs[R0_Index_Sample])[0]
    for i_house in cp.nonzero(Children0 * nneighbs)[0]:
        t2 = time.time()
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
                seed = 1
                # int.from_bytes(os.urandom(4),'big') # Generates random seed from OS
                Pregnancy_Newborns(grids, blocks, (N, Susceptible, seed, Pregnancy,
                                                   gestation, Children, cp.float32(ssigma)))

                seed = 1
                # int.from_bytes(os.urandom(4),'big') # Generates random seed from OS
                Recover_Infected(grids, blocks, (N, Infected, Recovered, seed,
                                                 Pincubtrans, AllInfected, ip))

                grids = (math.ceil(NT/blocksize_x), 1, 1)  # set grid size NT
                InfNeighb.fill(0)
                Infected_Neighbors(grids, blocks, (NT, Children_mtx_indx, AllInfected,
                                                   InfNeighb))

                grids = (math.ceil(N/blocksize_x), 1, 1)  # set grid size NT
                Pinfection_update(grids, blocks, (N, P_infection, InfNeighb, Children,
                                                  AllInfected, cp.float32(betarow), cp.float32(betahrow)))

                seed = 1
                # int.from_bytes(os.urandom(4),'big') # Generates random seed from OS
                New_Infected(grids, blocks, (N, P_infection, Infected, Susceptible,
                                             seed, AllInfected, ip))

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
        t3 = time.time()
        print("run time of infection for ", i_house, (t3-t2))
    t1 = time.time()
    time_diff = t1-t0
    print("total run time for index house: ", round((t1 - t0)))
    print(R0_Index_Sample)

    R0[row] = R0[row]/(r0_repeat*N_Index)
    print("row:", row, "Plink:", Plink[row], "beta:", betarow, "R0:", R0[row])


endtime = time.time()
# /(rowend+1-rowstart)))
print("seconds/network: ", round((endtime - starttime)))
