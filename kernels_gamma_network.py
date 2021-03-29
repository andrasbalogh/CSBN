import cupy as cp

gamma_setcpmtx = cp.RawKernel(r'''
#include <curand_kernel.h>             //needed for curand 
extern "C" __global__ void gamma_setcpmtx(const unsigned long int NPchunk, 
           const unsigned long int NPshift, const float Plink, 
           const float Pret, const float Padd, const lambdaTheta, int seed1, int seed2, const int* Children, 
           int* Children_mtx_chunk, int* Parents_mtx_chunk){
      unsigned long int k = blockDim.x * blockIdx.x + threadIdx.x;
      unsigned long int i, j, seq, offset;
      seed1=seed1+NPshift+k; seq = 0;  offset = 0; // different seed 
      float G = 0.0;

      curandState h;     // h1, h2 will store the random numbers

      curand_init(seed1,seq,offset,&h); // init random number generator

      if(k<NPchunk){  
         j=(int)floor(0.5*(1.0+sqrt(8.0*(NPshift+k)+1.0)));
         i=(int)(k+NPshift-(j-1)*j/2);
         if (Children[i]*Children[j] > 0) {
            G = 0;
            for(int l = 1; l < Children[i] + Children[j]; l++) {
               G = G + log(curand_uniform(&h));
            }
            if ( G < lambdaTheta) {
               Children_mtx_chunk[k]=1;
               if(curand_uniform(&h)<Pret){Parents_mtx_chunk[k]=1;}
               else{Parents_mtx_chunk[k]=0;}
            }
         } else {
            Children_mtx_chunk[k]=0;
            if(curand_uniform(&h1)<Padd){Parents_mtx_chunk[k]=1;}
               else{Parents_mtx_chunk[k]=0;}
         }
       } else {
          Children_mtx_chunk[k]=0;
            if(curand_uniform(&h1)<Padd){Parents_mtx_chunk[k]=1;}
               else{Parents_mtx_chunk[k]=0;}
       }

  }
  ''', 'gamma_setcpmtx', backend='nvcc') # have to use nvcc with curand_kernel