import cupy as cp

childrens_barabasi = cp.RawKernel(r'''
        #include <curand_kernel.h>
        extern "C" __global__
        void childrens_barabasi(const int j, int seed, const int Sumcd, int changed, const int* Children, int* deg, int* mtx_index){
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                int seq, offset, k;
                seq = 0;
                offset = 0;
                curandState h;
                if(i<j){ 
                curand_init(seed+i,seq,offset,&h);
                        if (curand_uniform(&h)<Children[i]*deg[i]/Sumcd){
                                k=i+(j*(j-1))/2;
                                mtx_index[i]=k;
                                deg[j]=deg[j]+1;
                                deg[i]=deg[i]+1;
                                changed=changed+Children[i]+Children[j];
                                }
                }
        }
''', 'children_barabasi', backend = 'nvcc')