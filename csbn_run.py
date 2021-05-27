from multiprocessing import Process, Lock
import os
import sys
import glob
from parameters import *  # parameter file
from fcsbn import *

# create data directory if it does not exist
if not os.path.exists('data'):
    os.makedirs('data')

# Have to delete previous files because results are added to the files line-by-line
if epidemic_run*epidemic_save>0: 
    # Get a list of all the files 
    fileList = glob.glob('data/epidemic-q-*.txt')
    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

runpergpu=(netindx_end-netindx_start+1)//gpuNum
runremainder= netindx_end-netindx_start+1-runpergpu*gpuNum            

if __name__ == '__main__':
    StartCurrent=netindx_start
    EndCurrent=StartCurrent
    for GPUindx in range(runremainder):
        EndCurrent=StartCurrent+runpergpu
        Process(target=fcsbn,args=(GPUindx,StartCurrent, EndCurrent,)).start()
        StartCurrent=StartCurrent+runpergpu+1
    if runpergpu>0:
        for GPUindx in range(runremainder,gpuNum):
            EndCurrent=StartCurrent+runpergpu-1
            Process(target=fcsbn,args=(GPUindx,StartCurrent, EndCurrent,)).start()
            StartCurrent=StartCurrent+runpergpu


