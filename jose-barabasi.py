import numpy as np
import matplotlib.pyplot as plt
import cupy as cp # CUDA accelerated library
import sys
import os # needed for reading random seed form OS file /dev/random
from kernels_barabasi import * # kernel functions (CUDA c++)
blocksize_x = 1024 # maximum size of 1D block is 1024 threads
Pc = 0.4
Padd=0.0004;  # parents' network probability of adding a connection if children did not have one
Pret=0.6    # parents' network probability of retaining children's connection
Mc = 7
N = 100000
Nsp_Children = 1000000
Nsp_Parents = 1000000

def barabasi (N, Nsp_Children, Nsp_Parents, Pc, Mc, Padd, Pret):

    N_mtx_Children = 0 # will store the actual number of connections
    N_mtx_Parents = 0 # will store the actual number of connections
    Children=cp.random.binomial(Mc, Pc, size = N, dtype=cp.int32)
    Children_mtx_index=cp.zeros(Nsp_Children, dtype = cp.int64)
    Parents_mtx_index=cp.zeros(Nsp_Parents, dtype = cp.int64)
    c_mtx_index=cp.zeros(N, dtype = cp.int64)
    p_mtx_index=cp.zeros(N, dtype = cp.int64)
    #Pd=cp.zeros(N, dtype = cp.float32)
    deg=cp.zeros(N, dtype = cp.int32)
    deg[0]=1
    deg[1]=1
    Sumcd=(Children[0]*deg[0]+Children[1]*deg[1]).get().item()
    N_mtx_Children=1
    N_mtx_Parents=1 # Should it be read and added with a probability ?????????
    #Pd[0]=Children[0]*deg[0]/Sumcd
    #Pd[1]=Children[1]*deg[1]/Sumcd
    blocks=(blocksize_x,1,1)    # number of blocks in the grid to cover all indices see grid later
    grids=(N//blocksize_x+ 1*(N % blocksize_x != 0),1,1) # set grid size
    for j in range(2,N):
        c_mtx_index.fill(0)
        p_mtx_index.fill(0)
        seed=int.from_bytes(os.urandom(4),'big') # Random seed from OS
        #grids=(j//blocksize_x+ 1*(j % blocksize_x != 0),1,1) # set grid size
        childrens_barabasi(grids, blocks, (j, seed, Sumcd, Children, deg, 
            c_mtx_index, p_mtx_index, cp.float32(Pret), cp.float32(Padd)))
        Sumcd=cp.dot(Children,deg).get().item()
        #Sumcd=Sumcd+changed
        nzc=cp.count_nonzero(c_mtx_index).get().item() # count how many nonzeros
        if (nzc>0):  # Children's connection matrix
            N_mtx_Children=N_mtx_Children+nzc
            if (N_mtx_Children>Nsp_Children):
                sys.exit("Error, must increase Nsp_Children!")
            cp.put(Children_mtx_index, cp.arange(N_mtx_Children-nzc,
                                stop=N_mtx_Children, step=1,
                                dtype=cp.int64),
                                c_mtx_index[cp.flatnonzero(c_mtx_index)])
        nzp=cp.count_nonzero(p_mtx_index).get().item() # count how many nonzeros
        if (nzp>0):  # Children's connection matrix
            N_mtx_Parents=N_mtx_Parents+nzp
            if (N_mtx_Parents>Nsp_Parents):
                sys.exit("Error, must increase Nsp_Parents!")
            cp.put(Parents_mtx_index, cp.arange(N_mtx_Parents-nzp,
                                stop=N_mtx_Parents, step=1,
                                dtype=cp.int64),
                                p_mtx_index[cp.flatnonzero(p_mtx_index)])
    from cupyx.scipy import sparse
    Jc = (cp.floor((1+cp.sqrt(8*Children_mtx_index[0:N_mtx_Children]+1))//2)).astype(cp.int64)
    Ic = (Children_mtx_index[0:N_mtx_Children] - ((Jc*(Jc-1))//2)).astype(cp.int64)# upper triangular part
    CP = sparse.coo_matrix((cp.ones(int(N_mtx_Children), dtype=cp.float32), (Ic, Jc)), 
                        shape=(N, N))
    sC= cp.ravel(sparse.spmatrix.sum(CP, axis=0)+cp.transpose(sparse.spmatrix.sum(CP, axis=1)))
    Jp = (cp.floor((1+cp.sqrt(8*Parents_mtx_index[0:N_mtx_Parents]+1))//2)).astype(cp.int64)
    Ip = (Parents_mtx_index[0:N_mtx_Parents] - ((Jp*(Jp-1))//2)).astype(cp.int64)
    PP = sparse.coo_matrix((cp.ones(int(N_mtx_Parents), dtype=cp.float32), (Ip, Jp)), shape=(N, N))
    sP= cp.ravel(sparse.spmatrix.sum(PP, axis=0)+cp.transpose(sparse.spmatrix.sum(PP, axis=1)))
    M=int(max(cp.amax(sC),np.amax(sP)))+1
    M=int(np.amax(sC))+1
    C_P_hist = plt.figure(1)
    plt.hist([sC.get(),sP.get()], bins=range(M+1), align='left',rwidth=0.9, label=["Children", "Parents"])
    #plt.hist(sC.get(), bins=range(M+1), align='left',rwidth=0.9, label=["Children"])
    plt.xlabel('Number of connections',fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.grid(True)
    plt.legend(prop={'size': 10})
    #        plt.title("$P_{link}=$"+str(Plink)+", $P_{ret}=$"+str(Pret)+", $P_{add}=$"+str(Padd)+", $N=$"+str(N))
    plt.title("$N=$"+str(N))
    plt.show()
    """plt.figure(2)
    import networkx as nx
    njc=Jc.get(); nic=Ic.get()
    G = nx.Graph()
    for i in range(N_mtx_Children):
        G.add_edge(nic[i], njc[i])
    nx.draw(G, with_labels=True) 
    plt.show()"""

barabasi (N, Nsp_Children, Nsp_Parents, Pc, Mc, Padd, Pret)