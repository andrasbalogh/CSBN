import cupy as cp
import math  # for ceiling function

setcpmtxR0 = cp.RawKernel(r'''
    #include <curand_kernel.h>             //needed for curand 
    extern "C" __global__ void setcpmtxR0(const int NTchunk, 
           int seed, float* Children_mtx_chunk){
        unsigned long int k = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned long int i, j, seq, offset;
        seed=seed+k+10; seq = 0;  offset = 0; // different seed
        curandState h;     // h will store the random numbers
        if(k<NTchunk){ 
            curand_init(seed,seq,offset,&h); // initiate random number generator 
            Children_mtx_chunk[k]= curand_uniform(&h);
        }
    }
    ''', 'setcpmtxR0', backend='nvcc') # have to use nvcc with curand_kernel.h

N=10000
temp = cp.zeros(N, cp.float32)
blocksize_x = 1024
blocks = (blocksize_x, 1, 1)
grids = (math.ceil(N/blocksize_x), 1, 1)
seed = 1

setcpmtxR0(grids,blocks,(N, seed, temp))
print(temp[500:510])