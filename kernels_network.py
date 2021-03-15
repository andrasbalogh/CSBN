import cupy as cp

setcpmtx = cp.RawKernel(r'''
#include <curand_kernel.h>             //needed for curand 
extern "C" __global__ void setcpmtx(const unsigned long int NPchunk, 
           const unsigned long int NPshift, const float Plink, 
           const float Pret, const float Padd, int seed, const int* Children, 
           int* Children_mtx_chunk, int* Parents_mtx_chunk){
      unsigned long int k = blockDim.x * blockIdx.x + threadIdx.x;
      unsigned long int i, j, seq, offset;
      seed=seed+NPshift+k; seq = 0;  offset = 0; // different seed
      curandState h;     // h will store the random numbers
      if(k<NPchunk){ 
         curand_init(seed,seq,offset,&h); // init random number generator 
         j=(int)floor(0.5*(1.0+sqrt(8.0*(NPshift+k)+1.0)));
         i=(int)(k+NPshift-(j-1)*j/2);
         if(curand_uniform(&h)<Plink*sqrt((float)(Children[i]*Children[j]))){
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
  ''', 'setcpmtx', backend='nvcc') # have to use nvcc with curand_kernel.h



trn_setcpmtx = cp.RawKernel(r'''
#include <curand_kernel.h>             //needed for curand 
extern "C" __global__ void trn_setcpmtx(const unsigned long int NPchunk, 
           const unsigned long int NPshift, const float Plink, 
           const float Pret, const float Padd, int seed1, int seed2, const int* Children, 
           int* Children_mtx_chunk, int* Parents_mtx_chunk){
      unsigned long int k = blockDim.x * blockIdx.x + threadIdx.x;
      unsigned long int i, j, seq, offset;
      seed1=seed1+NPshift+k; seq = 0;  offset = 0; // different seed 
      seed2=seed2+NPshift+k; seq = 0;  offset = 0; // second seed needed here or in csbn???

      curandState h1, h2;     // h1, h2 will store the random numbers

      curand_init(seed1,seq,offset,&h1); // init random number generator
      curand_init(seed2,seq,offset,&h2); // init random number generator

      if(k<NPchunk){  
         j=(int)floor(0.5*(1.0+sqrt(8.0*(NPshift+k)+1.0)));
         i=(int)(k+NPshift-(j-1)*j/2);
         
         // trn
         if((-Children[i]*(log(curand_uniform(&h1))))+(-Children[j]*(log(curand_uniform(&h2)))) > 40){
            Children_mtx_chunk[k]=1;
            if(curand_uniform(&h1)<Pret){Parents_mtx_chunk[k]=1;}
               else{Parents_mtx_chunk[k]=0;}
         }else{
            Children_mtx_chunk[k]=0;
            if(curand_uniform(&h1)<Padd){Parents_mtx_chunk[k]=1;}
               else{Parents_mtx_chunk[k]=0;}
         }
       }
  }
  ''', 'trn_setcpmtx', backend='nvcc') # have to use nvcc with curand_kernel.h