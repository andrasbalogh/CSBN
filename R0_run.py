####### imports ###########
from multiprocessing import Process, Lock
import os
import csv
from fR0 import *
###################################################

######## parameters ##################
Ngpu=8   # number of GPUs to distribute the processing of rows from R0-params.csv

N = 100000  # number of households
Nsp_Children = 20000000    # estimated upper bound for the storage of the children's network 

R0_samplesize = 10 # number of households to choose for sampling because all the households would take too long
R0_repeat = 1  # how many times to repeat the disease process for averaging for each household

# choosing rows from R0-params.csv data read: Plink, beta, h
# results are shown in terminal and saved in file R0-results.csv (overwritten at every run!)
rowstart = 1
rowend = 2

# Comment out the desired network
network_type = "ern_cpmtx" 
#network_type = "ban_cpmtx"           
#network_type = "trnGamma_cpmtx" 
#network_type = "trnExp_cpmtx"

lambdaTheta = -40.0 # parameter for trn #-20
Pc = 0.5  # 0.4 # Probability of having a child
Mc = 7  # Maximum number of children in a family initially
ssigma = 0.005    # Birth rate
gestation = 280
MaxDays = 100 # estimated upper bond for the length of the epidemic (for storage)

# The recovery process follows gamma distribution ip: days it takes to recover
# mu = 22; ip=28;  # Pertosis
mu = 11; ip = 16 # Measles

network_print = 1
epidemic_save = 0
epidemic_print = 0

####### Do not modify below!!!! #######

params=[ip, mu, N, Nsp_Children, Pc, Mc, network_print, lambdaTheta, network_type, R0_samplesize,
        ssigma, gestation, R0_repeat, MaxDays]

# create data directory if it does not exist
if not os.path.exists('data'):
    os.makedirs('data')

with open('R0-results.csv', 'w', newline='') as csvfile:  # overwrites the file
    r0writer = csv.writer(csvfile, delimiter=' ')
    r0writer.writerow(['row', ',',  'N', ',', 'Plink', ',', 'beta', ',', 'h', ',', 'R0'])

runpergpu=(rowend-rowstart+1)//Ngpu
runremainder= rowend-rowstart+1-runpergpu*Ngpu            

if __name__ == '__main__':
    StartCurrent=rowstart
    EndCurrent=StartCurrent
    for GPUindx in range(runremainder):
        EndCurrent=StartCurrent+runpergpu
        Process(target=fR0,args=(GPUindx,StartCurrent, EndCurrent, params, )).start()
        StartCurrent=StartCurrent+runpergpu+1
    for GPUindx in range(runremainder,Ngpu):
        EndCurrent=StartCurrent+runpergpu-1
        Process(target=fR0,args=(GPUindx,StartCurrent, EndCurrent, params, )).start()
        StartCurrent=StartCurrent+runpergpu

