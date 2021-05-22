#Distributes the work over several GPUs
import subprocess
import numpy as np
from parameters import *  # parameter file
import os
import sys

if epidemic_run*epidemic_save>0: # delete previous results
    for q in np.arange(qmin, qmax+dq, dq, dtype=float):
        filename="data/epidemics-q-{:02d}.txt".format(int(100*q))
        if os.path.exists(filename):
            os.remove(filename)

runpergpu=(NEnd_netindx-NStart_netindx+1)//gpuNum
runremainder= NEnd_netindx-NStart_netindx+1-runpergpu*gpuNum
runstr=''

StartCurrent=NStart_netindx
EndCurrent=StartCurrent
for i in range(runremainder):
    EndCurrent=StartCurrent+runpergpu
    # assemble the command line call in the format
    # export CUDA_VISIBLE_DEVICES="i";  python csbn.py startindex endindex
    runstr=runstr+'export CUDA_VISIBLE_DEVICES="{:1d}"; python3 csbn.py {:3d} {:3d} & '.format(i, StartCurrent, EndCurrent)
    StartCurrent=StartCurrent+runpergpu+1
if NEnd_netindx-NStart_netindx+1>gpuNum:
    for i in range(runremainder,gpuNum):
        EndCurrent=StartCurrent+runpergpu-1
        runstr=runstr+'export CUDA_VISIBLE_DEVICES="{:1d}"; python3 csbn.py {:3d} {:3d} & '.format(i, StartCurrent, EndCurrent)
        StartCurrent=StartCurrent+runpergpu

#start the running the processes as subprocesses in shell on different GPU
subprocess.run(runstr, shell=True)


