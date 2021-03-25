import cupy as cp

setcpmtxR0 = cp.RawKernel(r'''
    #include <curand_kernel.h>             //needed for curand 
    extern "C" __global__ void setcpmtxR0(const unsigned long int NTchunk, 
           const unsigned long int NTshift, const float Plink, 
           int seed, const int* Children, int* Children_mtx_chunk){
        unsigned long int k = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned long int i, j, seq, offset;
        seed=seed+NTshift+k; seq = 0;  offset = 0; // different seed
        curandState h;     // h will store the random numbers
        if(k<NTchunk){ 
            curand_init(seed,seq,offset,&h); // init random number generator 
            j=(int)floor(0.5*(1.0+sqrt(8.0*(NTshift+k)+1.0)));
            i=(int)(k+NTshift-(j-1)*j/2);
            if(curand_uniform(&h)<Plink*sqrt((float)(Children[i]*Children[j]))){
                Children_mtx_chunk[k]=1;}
            else{
                Children_mtx_chunk[k]=0; }
        }
    }
    ''', 'setcpmtxR0', backend='nvcc') # have to use nvcc with curand_kernel.h

pregnancy_burn_in = cp.RawKernel(r'''
    #include <curand_kernel.h>
    extern "C" __global__  
    void pregnancy_burn_in(const int N, const float ssigma, int seed, const int* Children, 
                   int* Pregnancy, const int gestation){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j, seq, offset;
        seq = 0;
        offset = 0;
        curandState h;
        if(i<N){ 
        curand_init(seed+i,seq,offset,&h);
            for(j=1; j < gestation; j++){
               if(Pregnancy[i]>0) { ++Pregnancy[i]; }
               else if( curand_uniform(&h) < ssigma/(1.0+exp(2.5*(Children[i]-2)))) { Pregnancy[i]=1; }
            }
        }
    }
     ''', 'pregnancy_burn_in', backend='nvcc')

Pregnancy_Newborns = cp.RawKernel(r'''
     #include<curand_kernel.h>
     extern "C" __global__  
     void Pregnancy_Newborns(const int N, int* Susceptible, const int seed, 
            int* Pregnancy, int gestation, int* Children, float ssigma){
         int i = blockDim.x * blockIdx.x + threadIdx.x;
         int seq, offset;
         seq = 0;
         offset = 0;
         curandState h;
         if(i<N){ 
            curand_init(seed+i,seq,offset,&h);
            if (Pregnancy[i]==0){
               if(curand_uniform(&h) < ssigma/(1.0+exp(2.5*(Children[i]-2.0)))){ Pregnancy[i]= 1; }
            }
            // pregnant for another day
            else if(Pregnancy[i]<gestation){ Pregnancy[i]=Pregnancy[i]+1; }
            // otherwise, newborn
            else if(Pregnancy[i] == gestation) { 
                Children[i]=Children[i]+1; 
                Pregnancy[i]=0; 
                Susceptible[i]= Susceptible[i]+1;
            }
         }
 }
     ''', 'Pregnancy_Newborns', backend='nvcc')

Recover_Infected = cp.RawKernel(r'''
     #include<curand_kernel.h>
     extern "C" __global__  
     void Recover_Infected(const int N, int* Infected, int* Recovered, const int seed, const float* Pincubtrans, int* AllInfected, const int ip){
         int i = blockDim.x * blockIdx.x + threadIdx.x;
         int seq, offset;
         seq = 0;
         offset = 0;
         curandState h;
         int dayofinf, k, nInf_i_day;
         if(i<N){ 
           curand_init(seed+i,seq,offset,&h);
           // all recoveres after incubation
           Recovered[i]=Recovered[i]+Infected[i*ip+ip-1];
           AllInfected[i]=AllInfected[i]-Infected[i*ip+ip-1];
           Infected[i*ip+ip-1]=0;
           for(dayofinf=ip-2; dayofinf>=0; dayofinf--) {
             // Inf(i,0) Inf(i,1)...Inf(i,ip-2) Inf(i,ip-1)
             nInf_i_day=Infected[i*ip+dayofinf];
             for(k=1; k<=nInf_i_day; k++) { 
               if(curand_uniform(&h)< Pincubtrans[dayofinf]) {
                 Recovered[i]=Recovered[i]+1; // one more recovers
                 Infected[i*ip+dayofinf]=Infected[i*ip+dayofinf]-1; // one less infected
                 AllInfected[i]=AllInfected[i]-1;
               }
             }
             Infected[i*ip+dayofinf+1]=Infected[i*ip+dayofinf]; // another day for infected
           }
           Infected[i*ip+0]=0; // there will be new infected later 
         }
 }
     ''', 'Recover_Infected', backend='nvcc')

Infected_Neighbors = cp.RawKernel(r'''
     extern "C" __global__  
     void Infected_Neighbors(const int NC, unsigned long int* Children_mtx_indx, int* AllInfected, int* InfNeighb){
         unsigned long int i, j;
         int k = blockDim.x * blockIdx.x + threadIdx.x;
         if(k<NC){ 
            j=(int)floor(0.5*(1.0+sqrt(8.0*Children_mtx_indx[k]+1.0)));
            i=(int)(Children_mtx_indx[k]-(j-1)*j/2);   
            InfNeighb[i]=InfNeighb[i]+AllInfected[j];
            InfNeighb[j]=InfNeighb[j]+AllInfected[i];
         }
     }
     ''', 'Infected_Neighbors')

Pinfection_update = cp.RawKernel(r'''
     extern "C" __global__  
     void Pinfection_update(const int N, float* P_infection, int* InfNeighb, int* Children, int* AllInfected, float bbeta, float bbetah){
         int i = blockDim.x * blockIdx.x + threadIdx.x;
         if(i<N){ 
           if(Children[i]>0) {
             P_infection[i]=1.0-powf(1.0-bbeta,(float)InfNeighb[i]/(float)Children[i])*powf(1.0-bbetah,AllInfected[i]); }
           else { P_infection[i]=0.0; }
         }
     }
     ''', 'Pinfection_update')

New_Infected = cp.RawKernel(r'''
     #include<curand_kernel.h>
     extern "C" __global__  
     void New_Infected(const int N, float* P_infection, int* Infected, int* Susceptible, const int seed, int* AllInfected, const int ip){
         int i = blockDim.x * blockIdx.x + threadIdx.x;
         int seq, offset;
         seq = 0;
         offset = 0;
         curandState h;
         int j;
         
         if(i<N){ 
           curand_init(seed+i,seq,offset,&h);
           for(j=0; j<Susceptible[i]; j++){
             if(curand_uniform(&h) < P_infection[i]) {
               //flattened array
               Infected[i*ip]=Infected[i*ip]+1;
               AllInfected[i]=AllInfected[i]+1;
               Susceptible[i]=Susceptible[i]-1;
             }
           }
         }
 }
     ''', 'New_Infected', backend='nvcc')



