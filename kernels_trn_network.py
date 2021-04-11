import cupy as cp

trnExp_cpmtx = cp.RawKernel(r'''
#include <curand_kernel.h>             //needed for curand 
extern "C" __global__ void trnExp_cpmtx(const unsigned long int NPchunk, 
           const unsigned long int NPshift, const float Plink, 
           const float Pret, const float Padd, const float lambdaTheta, int seed1, const int* Children, 
           int* Children_mtx_chunk, int* Parents_mtx_chunk){
      unsigned long int k = blockDim.x * blockIdx.x + threadIdx.x;
      unsigned long int i, j, seq, offset;
      seed1=seed1+NPshift+k; seq = 0;  offset = 0; // different seed 

      curandState h;     // h1 will store the random numbers

      curand_init(seed1,seq,offset,&h); // init random number generator

      if(k<NPchunk){  
         j=(int)floor(0.5*(1.0+sqrt(8.0*(NPshift+k)+1.0)));
         i=(int)(k+NPshift-(j-1)*j/2);
         
         // trn - change here:
         if(Children[i]*log(curand_uniform(&h)) + Children[j]*log(curand_uniform(&h)) < lambdaTheta 
         && Children[i]*Children[j] > 0) {
            Children_mtx_chunk[k]=1;
            if(curand_uniform(&h)<Pret){Parents_mtx_chunk[k]=1;}
               else{Parents_mtx_chunk[k]=0;}
         }else{
            Children_mtx_chunk[k]=0;
            if(curand_uniform(&h)<Padd){Parents_mtx_chunk[k]=1;}
               else{Parents_mtx_chunk[k]=0;}
         }
       }
  }
  ''', 'trnExp_cpmtx', backend='nvcc') # have to use nvcc with curand_kernel.h