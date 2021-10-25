import numpy as np
from csv import reader
import matplotlib.pyplot as plt
from parameters import *  # parameter file

qx=np.arange(qmin,qmax+dq,dq,dtype=float)
Nq=qx.size
qn=np.arange(0,Nq,1,dtype=int)
Nr=NQ*(netindx_end-netindx_start+1) # number of rows in the files data/epidemic-q-...csv

Ncol=11 # number of columns
# lastday, Sum(Daily_Incidence), sum(Daily_Vaccinators)), sum(Daily_Suscep), sum(Recovered),  maxloc(dIncidence), 
# maxval(dIncidence), Daily_Vaccinators(day), minval(Daily_Vaccinators(1:day)), maxval(Daily_Vaccinators(1:day), NChildren

A=np.zeros((Nr,Ncol,Nq),dtype=float)
stats=np.zeros((4,Ncol,Nq),dtype=float)


for q in qn:
    with open("data/epidemic-q-{:02d}.csv".format(int(qx[q]*100)), 'r') as inputfile:    
        rows = reader(inputfile)
        #rowi = next(rows);   rowi = next(rows); # two header lines
        for i in range(Nr):
            rowi = next(rows)   # keep these rows
            for j in range(Ncol):
                A[i,j,q] = float(rowi[j])
        stats[0:3,:,q]=np.percentile(A[:,:,q],[25, 75, 50],axis=0)
        stats[3,:,q]=np.mean(A[:,:,q],axis=0)
        stats[0,:,q]=stats[2,:,q]-stats[0,:,q]
        stats[1,:,q]=stats[1,:,q]-stats[2,:,q]
        
#print(stats[0:3,1,1])

namelabel = {0:'Duration of epidemic', 1:'Size of epidemic', 2:'Total number of vaccinated', 
    3:'Size of susceptible population', 4:'Size of recovered population', 5:'Day of the epidemic''s peak',
    6:'The peak of the epidemic', 7:'Vaccinators on last day', 8:'Minimum number of vaccinators',
    9:'Maximum number of vaccinators'} 
filename = { 0:'data/q-duration.pdf', 1:'data/q-epidemic.pdf', 2:'data/q-vaccinated.pdf', 3:'data/q-susceptibles.pdf', 
    4:'data/q-recovered.pdf', 5:'data/q-peak-day.pdf', 6:'data/q-peak.pdf', 7:'data/q-vaccinators-last-day.pdf',
    8:'data/q-vaccinators-min.pdf', 9:'data/q-vaccinators-max.pdf'}


for id in range(0,10):  
    # 0=day, 1=Sum(dIncidence), 2=sum(dVaccinated), 3=sum(Suscep), 4=sum(Rv), 5=maxloc(dIncidence), 
    # 6=maxval(dIncidence), 7=dVaccinators(day), 8=minval(dVaccinators), 9= maxval(dVaccinators)
    #print(namelabel[id])
    med=stats[2,id,:]
    ave=stats[3,id,:]
    L=stats[0,id,:] # lower half of the confidence interval
    U=stats[1,id,:] # upper half of the confidence interval
    fig,(ax1)=plt.subplots(1,1)
    ax1.errorbar(qx,med,np.vstack((L,U)),capsize=3, ecolor='blue',elinewidth=2,label='Quartile range') #,'DisplayName','Quartile range');
    ax1.scatter(qx,med, color='blue', marker='o', edgecolors='blue', facecolors='blue',label='Median')
    ax1.scatter(qx,ave, color='red', marker='o', edgecolors='red', facecolors='red',label='Mean')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(labels, loc='best')
    plt.xlabel('q',fontfamily='Times'); plt.ylabel(namelabel[id],fontfamily='Times');
    plt.savefig(filename[id])
    plt.close()  

 
