import cupy as cp

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

Vaccinators_Init = cp.RawKernel(r'''
     #include <curand_kernel.h>
     extern "C" __global__  
     void Vaccinators_Init(const int N, const float VProb, int seed, int* Vaccinator_yesnonever){
         int i = blockDim.x * blockIdx.x + threadIdx.x;
         int seq, offset;
         seq = 0;
         offset = 0;
         curandState h;
         if(i<N){ 
               curand_init(seed+i,seq,offset,&h);
               if(curand_uniform(&h)<VProb) { Vaccinator_yesnonever[i] = 1; }
               else { Vaccinator_yesnonever[i] = 0; }
         }
     }
     ''', 'Vaccinators_Init', backend='nvcc')

Vaccinators_Separate = cp.RawKernel(r'''
     extern "C" __global__  
     void Vaccinators_Separate(const int N, int* Vaccinator_yesnonever, int* Nbrvacc_yes, int* Nbrvacc_no ){
         int i = blockDim.x * blockIdx.x + threadIdx.x;
         if(i<N){ 
            if (Vaccinator_yesnonever[i]>0) {
                Nbrvacc_yes[i]=1;
                Nbrvacc_no[i]=0;
            }
            else {
                Nbrvacc_yes[i]=0;
                Nbrvacc_no[i]=1;
            }
         }
     }
     ''', 'Vaccinators_Separate')

Pressure_Update = cp.RawKernel(r'''
     extern "C" __global__  
     void Pressure_Update(const int NP, unsigned long int* Parents_mtx_indx, int* Vaccinator_yesnonever, int* Nbrvacc_yes, int* Nbrvacc_no ){
         unsigned long int i, j;
         int k = blockDim.x * blockIdx.x + threadIdx.x;
         if(k<NP){ 
            j=(int)floor(0.5*(1.0+sqrt(8.0*Parents_mtx_indx[k]+1.0)));
            i=(int)(Parents_mtx_indx[k]-(j-1)*j/2);   
            if (Vaccinator_yesnonever[j]>0) { Nbrvacc_yes[i]++; }
            else { Nbrvacc_no[i]++; }
            if (Vaccinator_yesnonever[i]>0) {Nbrvacc_yes[j]++; }
            else { Nbrvacc_no[j]++; }
         }
     }
     ''', 'Pressure_Update')

pv_info_update = cp.RawKernel(r'''
     extern "C" __global__  
     void pv_info_update(const int N, float* PV_info, float ps, float q, const int* Nbrvacc_yes, const int* Nbrvacc_no, const int* Vaccinator_yesnonever ){
         int i = blockDim.x * blockIdx.x + threadIdx.x;
         if(i<N){ 
            if (Vaccinator_yesnonever[i]<0) {
                PV_info[i] = 0.0;
            }
            else {
                float pq_yes = powf(q,Nbrvacc_yes[i]);
                float pMq_no = powf((1.0-q),Nbrvacc_no[i]);
                float pMq_yes= powf((1.0-q),Nbrvacc_yes[i]);
                float pq_no = powf(q,Nbrvacc_no[i]);
                PV_info[i] = ps*pq_yes*pMq_no/(ps*pq_yes*pMq_no + (1.0-ps)*pMq_yes*pq_no);
            }
         }
     }
     ''', 'pv_info_update')

Vaccinator_update = cp.RawKernel(r'''
     #include<curand_kernel.h>
     extern "C" __global__  
     void Vaccinator_update(const int N, float* PV_info, const int seed, int* Vaccinator_yesnonever ){
         int i = blockDim.x * blockIdx.x + threadIdx.x;
         int seq, offset;
         seq = 0;
         offset = 0;
         curandState h;
         if(i<N){ 
            curand_init(seed+i,seq,offset,&h);
            if (Vaccinator_yesnonever[i]>-1) {
              if(curand_uniform(&h)<PV_info[i]){Vaccinator_yesnonever[i]=1;}
              else { Vaccinator_yesnonever[i] = 0; }
            }
         }
     }
     ''', 'Vaccinator_update', backend='nvcc')

Vaccinate_Susceptibles = cp.RawKernel(r'''
     #include<curand_kernel.h>
     extern "C" __global__  
     void Vaccinate_Susceptibles(const int N, int* Vaccinated_new, int* Vaccinated, int* Vaccinator_yesnonever, int* Susceptible, const int seed, const float rho, const float Padv, int* Adverse, const float Peff){
         int i = blockDim.x * blockIdx.x + threadIdx.x;
         int seq, offset;
         seq = 0;
         offset = 0;
         curandState h;
         int j;
         if(i<N){ 
            curand_init(seed+i,seq,offset,&h);
            Vaccinated_new[i] = 0;
            if (Susceptible[i]>0 && Vaccinator_yesnonever[i]>0) {
               // want to vaccinate and there are unvaccinated
               for (j=1; j <= Susceptible[i]; j++) {
                   if(curand_uniform(&h)<rho) {
                        //vaccine is available (on individual basis)
                        Vaccinated_new[i]=Vaccinated_new[i]+1;
                        Vaccinated[i]=Vaccinated[i]+1; 
                        if (curand_uniform(&h)<Peff) {
                           // the vaccine is effective
                           Susceptible[i]= Susceptible[i]-1; }
                        if (curand_uniform(&h)<Padv) {
                           // adverse effect of the vaccine
                           Adverse[i]=Adverse[i]+1; }
                   }
               }
            }
         }
     }
     ''', 'Vaccinate_Susceptibles', backend='nvcc')

Pregnancy_Newborns = cp.RawKernel(r'''
     #include<curand_kernel.h>
     extern "C" __global__  
     void Pregnancy_Newborns(const int N, int* Vaccinated_new, int* Vaccinated, int* Vaccinator_yesnonever, int* Susceptible, const int seed, const float Padv, int* Adverse, const float Peff, int* Pregnancy, int gestation, int* Children, float ssigma){
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
                   if (Vaccinator_yesnonever[i]>0) {
                      Vaccinated_new[i]=Vaccinated_new[i]+1;
                      Vaccinated[i]=Vaccinated[i]+1;
                      // the vaccine is NOT effective
                      if(curand_uniform(&h)>Peff) { Susceptible[i]= Susceptible[i]+1; }
                      // adverse effect of the vaccine
                      if (curand_uniform(&h)<Padv){ // adverse effect
                         Adverse[i]=Adverse[i]+1;
                         Vaccinator_yesnonever[i] = -1;
                      }
                   }
                   else { Susceptible[i]= Susceptible[i]+1; }
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
             // Inf(i,0) Inf(i,1)...Inf(i,ip−2) Inf(i,ip−1)
             nInf_i_day=Infected[i*ip+dayofinf]
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
               Infected[i*ip]=Infected[i*ip]+1; // flattened matrix Nxip, new infected at [i,0]
               AllInfected[i]=AllInfected[i]+1;
               Susceptible[i]=Susceptible[i]-1;
             }
           }
         }
 }
     ''', 'New_Infected', backend='nvcc')

