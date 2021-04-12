import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import powerlaw
from scipy import sparse

netindx = 3  # check if the files is available !!

filename="data/csbn_network{:03d}.npz".format(netindx)
data=np.load(filename)
N=data['N'].item()    # Number of households

NC=data['NC'].item()  # Number of children connections, length of Children_mtx_indx
NP=data['NP'].item()  # Number of parent connections, length of Parents_mtx_indx
Plink=data['Plink'].item() # Probability that there is a connection between children's households
Pret=data['Pret'].item()   # Probability of parents retaining children's connections
Padd=data['Padd'].item()   # Probability of parents adding connections if the children don't have one
I0=data['I0'].item()    # Initial number of infected children
Children_mtx_indx=np.asarray(data['Children_mtx_indx'],dtype=np.int64) # Length=NC, int64 is required because the indexing of the upper triangular entries goes from 0 to (N-1)*(N-2)/2

# column and row indices of nonzero entries 
Jc = (np.floor((1+np.sqrt(8*Children_mtx_indx[0:NC]+1))//2)).astype(np.int64)
Ic = (Children_mtx_indx[0:NC] - ((Jc*(Jc-1))//2)).astype(np.int64)
# creates the sparse matrix using the row and column indices
CP = sparse.coo_matrix((np.ones(int(NC), dtype=np.float32), (Ic, Jc)), shape=(N, N))
# To get the degrees of the vertices (households) we add up the row entries.
# but since we only stored the upper trianguler part, we also need to add the column entries 
# in order to get the full matrix
C_degrees= np.ravel(sparse.spmatrix.sum(CP, axis=0)+np.transpose(sparse.spmatrix.sum(CP, axis=1)))

M=int(np.amax(C_degrees))+2
C_hist = plt.figure(1)
plt.hist(C_degrees, bins=range(M), align='left',rwidth=0.9, label=["Children"])
plt.xlabel('Number of connections',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.grid(True)
plt.legend(prop={'size': 10})
plt.title("$N=$"+str(N))

plt.figure(2)
G = nx.Graph()
for i in range(NC):
    G.add_edge(Ic[i], Jc[i])

# plot graph (does not give good visual for large N)
nx.draw(G, with_labels=True) 
#plt.show()


# People believe that real-life networks are either small-world networks, or their degrees distribution follows power law. 
#(either or but not both)

# check smallness https://networkx.org reference manual pg. 517-
#A graph is commonly classified as small-world if sigma>1.
#sigma(G, niter=100, nrand=10, seed=None) # default values
sigma=nx.sigma(G, niter=1, nrand=1)
print("sigma=",sigma)

#The small-world coefficient (omega) ranges between -1 and 1. 
# Values close to 0 means the G features small- world characteristics. 
# Values close to -1 means G has a lattice shape whereas values close to 1 means G is a random graph.
# omega(G, niter=100, nrand=10, seed=None) #default values
omega=nx.omega(G, niter=1, nrand=1, seed=None)
print("omega=",omega)

# powerlaw
# https://pythonhosted.org/powerlaw/
# https://stackoverflow.com/questions/49908014/how-can-i-check-if-a-network-is-scale-free

#should zeros excluded?
#C_degrees=C_degrees[np.flatnonzero(C_degrees)]

fit = powerlaw.Fit(C_degrees) 
plt.figure(3)
fig2 = fit.plot_pdf(color='b', linewidth=2)
fit.power_law.plot_pdf(color='g', linestyle='--', ax=fig2)
#plt.show()
R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
print ("R,p=", R, p)
# where R is the likelihood ratio between the two candidate distributions. 
# This number will be positive if the data is more likely in the first distribution, 
# but you should also check p < 0.05

plt.show()
