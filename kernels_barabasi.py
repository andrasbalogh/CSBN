import cupy as cp

childrens_barabasi = cp.RawKernel(r'''
        #include <curand_kernel.h>
        extern "C" __global__
        void childrens_barabasi(const unsigned long int j, int seed, 
			const int Sumcd, const int* Children, int* deg, 
			unsigned long int* c_mtx_index, unsigned long int* p_mtx_index, 
			const float Pret, const float Padd){
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            int seq, offset;
			unsigned long int k;
            seq = 0;
            offset = 0;
            curandState h;
            if(i<j){ 
                curand_init(seed+i,seq,offset,&h);
                if (curand_uniform(&h)<Children[i]*deg[i]/(float)Sumcd){
                    k=i+(j*(j-1))/2;
                    c_mtx_index[i]=k;
                    deg[j]=deg[j]+1;
                    deg[i]=deg[i]+1;
					if(curand_uniform(&h)<Pret){p_mtx_index[i]=k;}
				}
				else if(curand_uniform(&h)<Padd){p_mtx_index[i]=k;}
			}
        }
''', 'childrens_barabasi', backend = 'nvcc')