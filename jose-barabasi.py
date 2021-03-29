import numpy as np
import matplotlib.pyplot as plt
import cupy as cp # CUDA accelerated library
import sys
import os # needed for reading random seed form OS file /dev/random
from kernels_barabasi import * # kernel functions (CUDA c++)
blocksize_x = 1024 # maximum size of 1D block is 1024 threads
Pc = 0.4
Mc = 7
N = 1000
Nsp_Children = 1000000

def barabasi (N, Nsp_Children, Pc, Mc):

    N_mtx_Children = 0 # will store the actual number of connections
    Children=cp.random.binomial(Mc, Pc, size = N)
    Children_mtx_index=cp.zeros(Nsp_Children, dtype = cp.int64)
    mtx_index=cp.zeros(N, dtype = cp.int64)
    Pd=cp.zeros(N, dtype = cp.float)
    deg=cp.zeros(N, dtype = cp.int64)
    deg[0]=1
    deg[1]=1
    Sumcd=(Children[0]*deg[0]+Children[1]*deg[1])
    N_mtx_Children=1
    #Pd[0]=Children[0]*deg[0]/Sumcd
    #Pd[1]=Children[1]*deg[1]/Sumcd
    blocks=(blocksize_x,1,1)    # number of blocks in the grid to cover all indices see grid later
    grids=(N//blocksize_x+ 1*(N % blocksize_x != 0),1,1) # set grid size
    for j in range(2,N):
        changed=0
        mtx_index.fill(0)
        seed=int.from_bytes(os.urandom(4),'big') # Random seed from OS
        childrens_barabasi(grids, blocks, (j, seed, Sumcd, changed, Children, deg, mtx_index))
        nzc=np.asscalar(cp.count_nonzero(mtx_index).get()) # count how many nonzeros
        if (nzc>0):  # Children's connection matrix
            N_mtx_Children=N_mtx_Children+nzc
            if (N_mtx_Children>Nsp_Children):
                sys.exit("Error, must increase Nsp_Children!")
            cp.put(Children_mtx_index, cp.arange(N_mtx_Children-nzc,
                                                stop=N_mtx_Children, step=1,
                                                dtype=cp.int64),
                                                cp.flatnonzero(mtx_index))
        """
        for i in range(0,j):
            if (np.random.uniform()<Children[i]*deg[i]/Sumcd):
                N_mtx_Children=N_mtx_Children+1
                k=i+(j*(j-1))//2
                Children_mtx_index[N_mtx_Children]=k
                deg[j]=deg[j]+1
                deg[i]=deg[i]+1
                changed=changed+Children[i]+Children[j]
                #print(changed)
        Sumcd=Sumcd+changed"""
    #print(deg)
    #print(Sumcd)
    #print(Children_mtx_index)
    from scipy import sparse
    Jc = (np.floor((1+np.sqrt(8*Children_mtx_index[0:N_mtx_Children]+1))/2)).astype(np.int64)
    Ic = (Children_mtx_index[0:N_mtx_Children] - ((Jc*(Jc-1))/2)).astype(np.int64)# upper triangular part
    CP = sparse.coo_matrix((np.ones(int(N_mtx_Children), dtype=np.float32), (Ic, Jc)), 
                        shape=(N, N))
    sC= np.ravel(sparse.spmatrix.sum(CP, axis=0)+np.transpose(sparse.spmatrix.sum(CP, axis=1)))
    #Jp = (cp.floor((1+cp.sqrt(8*Parents_mtx_indx[0:N_mtx_Parents]+1))/2)).astype(cp.int64)
    #Ip = (Parents_mtx_indx[0:N_mtx_Parents] - ((Jp*(Jp-1))/2)).astype(cp.int64)
    #PP = sparse.coo_matrix((cp.ones(int(N_mtx_Parents), dtype=cp.float32), (Ip, Jp)), shape=(N, N))
    #sP= cp.ravel(sparse.spmatrix.sum(PP, axis=0)+cp.transpose(sparse.spmatrix.sum(PP, axis=1)))
    #M=int(max(cp.amax(sC),np.amax(sP)))+1
    M=int(np.amax(sC))+1
    C_P_hist = plt.figure(1)
    #        plt.hist([sC.get(),sP.get()], bins=range(M+1), align='left',rwidth=0.9, label=["Children", "Parents"])
    plt.hist(sC, bins=range(M+1), align='left',rwidth=0.9, label=["Children"])
    plt.xlabel('Number of connections',fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.grid(True)
    plt.legend(prop={'size': 10})
    #        plt.title("$P_{link}=$"+str(Plink)+", $P_{ret}=$"+str(Pret)+", $P_{add}=$"+str(Padd)+", $N=$"+str(N))
    plt.title("$N=$"+str(N))
    plt.show()
    """plt.figure(2)
    import networkx as nx
    njc=Jc; nic=Ic
    G = nx.Graph()
    for i in range(N_mtx_Children):
        G.add_edge(nic[i], njc[i])
    nx.draw(G, with_labels=True) 
    plt.show()"""


barabasi (N, Nsp_Children, Pc, Mc)