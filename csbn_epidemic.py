import numpy as np # numerical library
import cupy as cp # CUDA accelerated library
import os  # for reading random seed from OS
import sys # for stopping code if needed
from scipy import stats
import matplotlib.pyplot as plt
from kernels_epidemic import * # kernel functions (CUDA c++)
import math # for ceiling function
from cupyx.scipy import sparse
from parameters import *  # parameter file

def csbn_epidemic(q, netindx):
    # Reading network data of children and parents
    filename="data/csbn_network{:03d}.npz".format(netindx)
    try:
        data=np.load(filename)
    except:
        print("Error, file", filename, " is missing! Make sure to ran the code with network_run=1")
        sys.exit()
    Ntemp=np.asscalar(data['N'])    # Number of households
    if(Ntemp!=N): 
        print("Error, N=",N," specified does not match N=",
                           Ntemp, " from data file!")
        sys.exit()
    NC=np.asscalar(data['NC'])  # Number of children connections, length of Children_mtx_indx
    NP=np.asscalar(data['NP'])  # Number of parent connections, length of Parents_mtx_indx
    Plink=np.asscalar(data['Plink']) # Probability that there is a connection between children's households
    Pret=np.asscalar(data['Pret'])   # Probability of parents retaining children's connections
    Padd=np.asscalar(data['Padd'])   # Probability of parents adding connections if the children don't have one
    I0temp=np.asscalar(data['I0'])    # Initial number of infected children
    if(I0temp!=I0): 
        print("Error, I0=",I0," specified does not match I0=",I0temp, " from data file!")
        sys.exit()
    Children_mtx_indx=cp.asarray(data['Children_mtx_indx'],dtype=cp.int64) # Length=NC, int64 is required because the indexing of the upper triangular entries goes from 0 to (N-1)*(N-2)/2
    Parents_mtx_indx=cp.asarray(data['Parents_mtx_indx'],dtype=cp.int64)   # Length=NP
    AllInfected=cp.asarray(data['Infected'],dtype=cp.int32)     # Total number of infected children in each household, Length=N, the standard size of integers is 32 bits. Decreasing the int size could complicate the code but could reduce the storage requirements.
    Susceptible=cp.asarray(data['Susceptible'],dtype=cp.int32)  # Total number of susceptible children in each household, Length=N    
    Children=cp.add(AllInfected,Susceptible)  # Total number of children. Later newborns are added
    nS=cp.asarray(data['nS'],dtype=cp.int32)
    
    # The recovery process follows gamma distribution ip: max days it takes to recover, mu: average length
    disc = np.arange(ip+1)
    cumprob = stats.gamma.cdf(disc, mu, scale=1)
    Pincubtrans = cp.asarray((cumprob[1:ip+1]-cumprob[0:ip])/(cumprob[ip]-cumprob[0:ip]), dtype=cp.float32)

    if (delta==9999):
        if(q-qeps<0 or q+qeps>1):
            print("Error! (q-qeps,q+qeps)=(%5.3f,%5.3f) is not in the [0,1] range!"% (q-qeps, q+qeps))
            sys.exit()

        if qeps==0.0:
            qij=q*cp.ones(NP, dtype=cp.float32)
            qji=q*cp.ones(NP, dtype=cp.float32)
        else:
            qij=cp.random.uniform(low=q-qeps, high=q+qeps, size=NP, dtype=cp.float32)
            qji=cp.random.uniform(low=q-qeps, high=q+qeps, size=NP, dtype=cp.float32)
    
    Pq_yes=cp.ones(N, dtype=cp.float32)
    P1q_yes=cp.ones(N, dtype=cp.float32)
    Pq_no=cp.ones(N, dtype=cp.float32)
    P1q_no=cp.ones(N, dtype=cp.float32)
    
    Infected=cp.zeros((N,ip),dtype=cp.int32)    
    Pregnancy=cp.zeros(N, dtype=cp.int32)               # Days of pregnancy in each household[0,gestation]
    Vaccinator_yesnonever=cp.zeros(N, dtype=cp.int32)   # Household vaccination: -1 = never, 0 = no, 1 = yes, 0/1 can change
    Adverse=cp.zeros(N, dtype=cp.int32)                 # Number of adverse effects in each household, newborns vs others?
    Vaccinated_new=cp.zeros(N, dtype=cp.int32)          # Number of daily vaccinated children per household
    Vaccinated=cp.zeros(N, dtype=cp.int32)              # Number of vaccinated children (cumulative in each household)
    Recovered=cp.zeros(N, dtype=cp.int32)               # Number of recovered children (cumulative in each household)
    InfNeighb=cp.zeros(N, dtype=cp.int32)               # Number of infected children of the neighbors
    Infected_Total=cp.zeros(MaxDays, dtype=cp.int32)    # Number of total infected children on a given day
    Daily_Incidence=cp.zeros(MaxDays, dtype=cp.int32)   # Number of newly infected children (incidence)
    Daily_Vaccinations=cp.zeros(MaxDays, dtype=cp.int32)  # Number of vaccinations on a given day
    Daily_Suscep=cp.zeros(MaxDays, dtype=cp.int32)        # Number of susceptible children on a given day
    Daily_Vaccinators=cp.zeros(MaxDays, dtype=cp.int32)   # Number of vaccinator households on a given day
    Daily_NonVaccinators=cp.zeros(MaxDays, dtype=cp.int32)# Number of non-vaccinator households on a given day
    Daily_Recovered=cp.zeros(MaxDays, dtype=cp.int32)     # Number of recovered children (total cumulative)
    Daily_P0=cp.zeros(MaxDays, dtype=cp.float32)          # Probability to vaccinate without any social influence
    Daily_Children=cp.zeros(MaxDays, dtype=cp.int32)      # Total number of children on each day
    PV_info=cp.zeros(N, dtype=cp.float32)      # Probability to vaccinate considering social influence
    P_infection=cp.zeros(N, dtype=cp.float32)  # Probability of infection based on number of infected children in the neighborhood and in the household 
    Infected[:,1]=AllInfected                  # Newly infected children on the first day
    nV=cp.zeros(N, dtype=cp.int32)             # number of vaccinating neighbors   
    grids=(math.ceil(N/blocksize_x),1,1) # set grid size
    seed=int.from_bytes(os.urandom(4),'big') # Random seed from OS
    # Uses the number of existing children and birth rate (ssigma) to create pregnancies at different stages in each household.
    # Pregnancy[i] = 0 - no pregnancy
    # Pregnancy[i] = j, 0 < j < gestation - jth day of pregnancy
    pregnancy_burn_in(grids, blocks, (N, cp.float32(ssigma), seed, Children, Pregnancy, gestation))

    # Initial set up of vaccinators and non-vaccinators, based on the initial infected (I0)
    # Vaccinator_yesnonever[i] = 0/1, family i doesn't vaccinate/does vaccinate. Can change, see Vaccinator_update 
    Vaccinator_yesnonever =(cp.random.rand(N)< (1.0/(1.0+np.exp(-aalpha*I0)))).astype(cp.int32)

    # Setting up never-vaccinators, Vaccinator_yesnonever[i] = -1. NV0 proportion of non-vaccinators
    Nvacc=np.asscalar(cp.count_nonzero(Vaccinator_yesnonever).get()) # Number of vaccinators
    N_nevervacc=int(round(NV0*(N-Nvacc))) # Calculates the number of never-vaccinators
    idx_Nonvacc=cp.argmin(Vaccinator_yesnonever) # Indices of non-vaccinators, min value of Vaccinator_yesnonever is 0
    idx_Nevervacc=cp.random.choice(N-Nvacc, size=N_nevervacc, replace=False, p=None) # From the number of non-vaccinators (N-Nvacc), chooserandomly N_nevervacc indices (never vaccinators)
    cp.put(Vaccinator_yesnonever, idx_Nevervacc, -(cp.ones(N_nevervacc,dtype=cp.int16))) # Set the never-vaccinators (-1) into Vaccinator_yesnonever
    
    Attr1=(cp.random.rand(N)< P_attr).astype(cp.int32)

    # Daily Process: Begins by initializing Day 0
    day=0
    Infected_Total[day] = cp.sum(AllInfected)
    Daily_Incidence[day] = cp.sum(Infected[:,1])
    Daily_Vaccinations[day] = cp.sum(Vaccinated_new)
    Daily_Vaccinators[day] = cp.sum(Vaccinator_yesnonever > 0)
    Daily_NonVaccinators[day] = cp.sum(Vaccinator_yesnonever <= 0)
    Daily_Suscep[day] = cp.sum(Susceptible)
    Daily_Recovered[day] = cp.sum(Recovered)
    Daily_Children[day] = cp.sum(Children)
    Daily_P0[day] = 1.0/(1.0+np.exp(ggamma*np.asscalar(cp.sum(Adverse).get())-aalpha*np.asscalar(cp.sum(Infected_Total).get())))
    while (Infected_Total[day].get() > 0):
        day = day + 1
        if (day>=MaxDays): 
            print("Error, must increase MaxDays!")
            sys.exit()

        # pr is the (global) probability to vaccinate based on total adverse effects and total infection, without social influence
        pi=-ggamma*np.asscalar(cp.sum(Adverse).get())+aalpha*np.asscalar(cp.sum(Infected_Total).get())
        pr= 1.0/(1.0+np.exp(-pi))

        if (delta==9999):  # qij or attributes are used
            Pq_yes.fill(1.0)
            P1q_yes.fill(1.0)
            Pq_no.fill(1.0)
            P1q_no.fill(1.0)
            grids=(math.ceil(NP/blocksize_x),1,1) # set grid size NP
            if(N_attr1<2): # qij is used
                Pressure_Update(grids, blocks, (NP, Parents_mtx_indx, Vaccinator_yesnonever, 
                                Pq_yes, P1q_yes, Pq_no, P1q_no, qij, qji))
            else:  # attributes are used
                seed=int.from_bytes(os.urandom(4),'big') # Random seed from OS
                Pressure_Updated_Attr1(grids, blocks, (NP, Parents_mtx_indx, Vaccinator_yesnonever, 
                                Pq_yes, P1q_yes, Pq_no, P1q_no, Attr1, seed))

            grids=(math.ceil(N/blocksize_x),1,1) # set grid size N
            # Updates PV_info - the probability to vaccinate based on social influence for each household
            # Social influence is based on the number of neighbors that vaccinate or do not vaccinate
            pv_info_update(grids, blocks, (N, PV_info, cp.float32(pr), Vaccinator_yesnonever, 
                                                Pq_yes, P1q_yes, Pq_no, P1q_no))
        else: # voting model
            nV.fill(0)
            grids=(math.ceil(NP/blocksize_x),1,1) # set grid size NP
            nV_Update(grids, blocks, (NP, Parents_mtx_indx, Vaccinator_yesnonever, nV))
            grids=(math.ceil(N/blocksize_x),1,1) # set grid size N
            # Updates PV_info - the probability to vaccinate based on social influence for each household
            pv_delta_update(grids, blocks, (N, PV_info, cp.float32(delta), cp.float32(pi), 
                                                 cp.float32(pr), nV, nS))

        # Of the households that may vaccinate, update based on PV_info. Never-vaccinators will not change
        seed=int.from_bytes(os.urandom(4),'big') # Generates random seed from OS
        Vaccinator_update(grids, blocks, (N, PV_info, seed, Vaccinator_yesnonever))

        # Vaccinates susceptibles if the household is a vaccinator and the vaccine is available
        # The vaccine is effective with probability Peff and causes adverse effects with probability Padv 
        seed=int.from_bytes(os.urandom(4),'big') # Generates random seed from OS
        Vaccinated_new.fill(0) # Resets the number of newly vaccinated children
        Vaccinate_Susceptibles(grids, blocks, (N, Vaccinated_new, Vaccinated, Vaccinator_yesnonever, Susceptible, seed, cp.float32(rho),cp.float32(Padv), Adverse, cp.float32(Peff)))
    
        seed=int.from_bytes(os.urandom(4),'big') # Generates random seed from OS
        Pregnancy_Newborns(grids, blocks, (N, Vaccinated_new, Vaccinated, Vaccinator_yesnonever, Susceptible, seed, cp.float32(Padv), Adverse, cp.float32(Peff), Pregnancy, gestation, Children, cp.float32(ssigma)))
    
        seed=int.from_bytes(os.urandom(4),'big') # Generates random seed from OS
        Recover_Infected(grids, blocks, (N, Infected, Recovered, seed, Pincubtrans, AllInfected, ip))
    
        grids=(math.ceil(NC/blocksize_x),1,1) # set grid size NC
        InfNeighb.fill(0)
        Infected_Neighbors(grids, blocks, (NC, Children_mtx_indx, AllInfected, InfNeighb))
    
        grids=(math.ceil(N/blocksize_x),1,1) # set grid size
        Pinfection_update(grids, blocks, (N, P_infection, InfNeighb, Children, AllInfected, cp.float32(bbeta), cp.float32(bbetah)))
    
        seed=int.from_bytes(os.urandom(4),'big') # Generates random seed from OS
        New_Infected(grids, blocks, (N, P_infection, Infected, Susceptible, seed, AllInfected, ip))
    
        Infected_Total[day] = cp.sum(AllInfected)
        Daily_Incidence[day] = cp.sum(Infected[:,1])
        Daily_Vaccinations[day] = cp.sum(Vaccinated_new)
        Daily_Vaccinators[day] = cp.sum(Vaccinator_yesnonever > 0) #reduction kernel
        Daily_NonVaccinators[day] = cp.sum(Vaccinator_yesnonever <= 0)
        Daily_Suscep[day] = cp.sum(Susceptible)
        Daily_Recovered[day] = cp.sum(Recovered)
        Daily_Children[day] = cp.sum(Children)
        Daily_P0[day] = pr
        #print("Day= ",day, " MaxP_infection= ", cp.amax(P_infection), " Infected_Total= ", Infected_Total[day])
    if epidemic_print:
        Days=np.arange(day+1)
        fig = plt.figure()
        fig.subplots_adjust(hspace=1.5, wspace=1)
        ax=fig.add_subplot(3,3,1)
        ax.plot(Days,Infected_Total[0:day+1].get())
        ax.set_title("Daily Infected")
        ax=fig.add_subplot(3,3,2)
        ax.plot(Days,Daily_Incidence[0:day+1].get())
        ax.set_title("Daily Incidence")
        ax=fig.add_subplot(3,3,3)
        ax.plot(Days,Daily_Vaccinations[0:day+1].get())
        ax.set_title("Daily Vaccinations")
        ax=fig.add_subplot(3,3,4)
        ax.plot(Days,Daily_Vaccinators[0:day+1].get())
        ax.set_title("Daily Vaccinators")
        ax=fig.add_subplot(3,3,5)
        ax.plot(Days,Daily_NonVaccinators[0:day+1].get())
        ax.set_title("Daily NonVaccinators")
        ax=fig.add_subplot(3,3,6)
        ax.plot(Days,Daily_Suscep[0:day+1].get())
        ax.set_title("Daily Suscep")
        ax=fig.add_subplot(3,3,7)
        ax.plot(Days,Daily_Recovered[0:day+1].get())
        ax.set_title("Daily Recovered")
        ax.set_xlabel("Days")
        ax=fig.add_subplot(3,3,8)
        ax.plot(Days,Daily_Children[0:day+1].get())
        ax.set_title("Daily Children")
        ax.set_xlabel("Days")
        ax=fig.add_subplot(3,3,9)
        ax.plot(Days,Daily_P0[0:day+1].get())
        ax.set_title("Daily P0") 
        ax.set_xlabel("Days")
        if (delta==9999):
            filename="data/epidemic-q-{:02d}-network-{:03d}.pdf".format(int(100*q),netindx)
        else:
            filename="data/epidemic-delta-network-{:03d}.pdf".format(netindx)
        fig.savefig(filename)
        plt.close()
    if(epidemic_save):
        if (delta==9999):
            filename=open("data/epidemic-q-{:02d}.csv".format(int(100*q)),"a+")
            file_infected="data/infected-q-{:02d}-network-{:03d}.csv".format(int(100*q), netindx)
        else:
            filename=open("data/epidemic-delta.csv","a+")
            file_infected="data/infected-delta-network-{:03d}.csv".format(netindx)
        #% day, Sum(Daily_Incidence), sum(Daily_Vaccinations)), sum(Daily_Suscep), sum(Recovered),  
        # maxloc(dIncidence), maxval(dIncidence), Daily_Vaccinators(day), 
        # minval(Daily_Vaccinators(1:day)), maxval(Daily_Vaccinators(1:day), NChildren
        filename.writelines("{:4d}, {:6d}, {:6d}, {:4d}, {:6d}, {:5d}, {:6d}, {:6d}, {:6d}, {:6d}, {:6d} \n".format(
            day+1, cp.sum(Daily_Incidence[0:day+1]).get(), cp.sum(Daily_Vaccinations[0:day+1]).get(), 
            cp.sum(Daily_Suscep[0:day+1]).get(), cp.sum(Daily_Recovered[0:day+1]).get(), 
            cp.argmax(Daily_Incidence[0:day+1]).get(), cp.amax(Daily_Incidence[0:day+1]).get(),
            Daily_Vaccinators[day].get(), cp.amin(Daily_Vaccinators[0:day+1]).get(), 
            cp.amax(Daily_Vaccinators[0:day+1]).get(), Daily_Children[day].get() ))
        filename.close()
        np.savetxt(file_infected, Infected_Total[0:day+1].get(), fmt='%i', delimiter=",")

