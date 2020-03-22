#include"spqlios++.hpp"
#include <random>
#include <cassert>
#include <iostream>
#include <cmath>

using namespace TFHEpp;
using namespace std;


template<typename T = uint32_t,uint32_t N = DEF_N>
__device__ inline void TwistMulInvert(double* res, const T* a, const double* twist){
    const unsigned int tid = threadIdx.x;
    const unsigned int bdim = blockDim.x;
        for (int i = tid; i < N / 2; i+=bdim) {
            const double are = static_cast<double>(static_cast<typename make_signed<T>::type>(a[i]));
            const double aim = static_cast<double>(static_cast<typename make_signed<T>::type>(a[i+N/2]));
            const double aimbim = aim * twist[i + N / 2];
            const double arebim = are * twist[i + N / 2];
            res[i] = are * twist[i] - aimbim;
            res[i + N / 2] = aim * twist[i] + arebim;
        }
        __threadfence();
    }

template<uint32_t N>
__device__ inline void ButterflyAdd(double* const a, double* const b){
    double& are = *a;
    double& aim = *(a+N);
    double& bre = *b;
    double& bim = *(b+N);
        
    const double tempre = are;
    are += bre;
    bre = tempre - bre;
    const double tempim = aim;
    aim += bim;
    bim = tempim - bim;
}

template <uint32_t Nbit = DEF_Nbit-1, uint32_t stride, bool isinvert =true>
__device__ inline void TwiddleMul(double* const a, const double* table,const int i,const int step){
    constexpr uint32_t N = 1<<Nbit;

    double& are = *a;
    double& aim = *(a+N);
    const double bre = table[stride*(1<<step)*i];
    const double bim = isinvert?table[stride*(1<<step)*i + N]:-table[stride*(1<<step)*i + N];

    const double aimbim = aim * bim;
    const double arebim = are * bim;
    are = are * bre - aimbim;
    aim = aim * bre + arebim;
}

template <uint32_t Nbit = DEF_Nbit-1, bool isinvert =true>
__device__ inline void Radix4TwiddleMul(double* const a){
    constexpr uint32_t N = 1<<Nbit;

    double& are = *a;
    double& aim = *(a+N);
    const double temp = are;
    are = aim;
    aim = temp;
    if constexpr(isinvert){
        are*=-1;
    }
    else{
        aim*=-1;
    }
}

template<uint32_t Nbit = DEF_Nbit-1>
__device__ inline void IFFT(double* const res, const double* table){
    constexpr uint32_t N = 1<<Nbit;
    constexpr uint32_t basebit  = 2;

    const unsigned int tid = threadIdx.x;
    const unsigned int bdim = blockDim.x;

    for(int step = 0; step<Nbit-1; step+=basebit){
        const uint32_t size = 1<<(Nbit-step);
        const uint32_t elementmask = (size>>basebit)-1;
        for(int index=tid;index<(N>>basebit);index+=bdim){
            const uint32_t elementindex = index&elementmask;
            const uint32_t blockindex = (index - elementindex)>>(Nbit-step-basebit);
            double* const res0 = &res[blockindex*size+elementindex];
            double* const res1 = res0+size/4;
            double* const res2 = res0+size/2;
            double* const res3 = res0+3*size/4;

            ButterflyAdd<N>(res0,res2);
            ButterflyAdd<N>(res1,res3);
            Radix4TwiddleMul<Nbit,true>(res3);

            ButterflyAdd<N>(res0,res1);
            ButterflyAdd<N>(res2,res3);

            TwiddleMul<Nbit,2,true>(res1,table,elementindex,step);
            TwiddleMul<Nbit,1,true>(res2,table,elementindex,step);
            TwiddleMul<Nbit,3,true>(res3,table,elementindex,step);
        }
        __threadfence();
    }
    constexpr uint32_t flag = Nbit%2;
    switch(flag){
        case 0:
            for(int index=tid;index<N/4;index+=bdim){
                double* const res0 = &res[index*4];
                double* const res1 = res0+1;
                double* const res2 = res0+2;
                double* const res3 = res0+3;

                ButterflyAdd<N>(res0,res2);
                ButterflyAdd<N>(res1,res3);
                Radix4TwiddleMul<Nbit,true>(res3);

                ButterflyAdd<N>(res0,res1);
                ButterflyAdd<N>(res2,res3);
            }
            __threadfence();
            break;
        case 1:
            for(int index=tid;index<N/2;index+=bdim){
                double* const res0 = &res[index*2];
                double* const res1 = res0+1;

                ButterflyAdd<N>(res0,res1);
            }
            __threadfence();
            break;
    }
}

__global__ void TwistIFFTlvl1(double* const res, const uint32_t* a, const double* const twistlvl1, const double* tablelvl1){
        __shared__ double buff[DEF_N];
        TwistMulInvert<uint32_t,DEF_N>(buff,a,twistlvl1);
        IFFT<DEF_Nbit-1>(buff,tablelvl1);
        for(int i = 0;i<DEF_N;i++) res[i] = buff[i];
        __threadfence();
    }

int main( int argc, char** argv) 
{
    const array<double,DEF_N> h_twistlvl1 = SPQLIOSpp::TwistGen<DEF_N>();
    const array<double,DEF_N> h_tablelvl1 = SPQLIOSpp::TableGen<DEF_N/2>();
    double* twistlvl1,*tablelvl1;
    cudaMalloc( (void**) &twistlvl1, sizeof(h_twistlvl1));
    cudaMalloc( (void**) &tablelvl1, sizeof(h_tablelvl1));
    cudaMemcpy( twistlvl1, h_twistlvl1.data(), sizeof(h_twistlvl1),cudaMemcpyHostToDevice);
    cudaMemcpy( tablelvl1, h_tablelvl1.data(), sizeof(h_tablelvl1),cudaMemcpyHostToDevice);

    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint32_t> Torus32dist(0, UINT32_MAX);

    Polynomiallvl1 a;
    for (uint32_t &i : a) i = Torus32dist(engine);

    PolynomialInFDlvl1 h_res,res;
    SPQLIOSpp::TwistIFFTlvl1(h_res,a);

    uint32_t* d_a;
    double* d_res;
    cudaMalloc( (void**) &d_a, sizeof(a));
    cudaMalloc( (void**) &d_res, sizeof(res));
    cudaMemcpy(d_a,a.data(),sizeof(a),cudaMemcpyHostToDevice);
    TwistIFFTlvl1<<<1,64>>>(d_res,d_a,twistlvl1,tablelvl1);
    cudaMemcpy(res.data(),d_res,sizeof(res),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i = 0;i<DEF_N;i++) {assert(abs(res[i]-h_res[i])<1e-3);}
    cudaFree(d_a);
    cudaFree(d_res);
    cudaFree(twistlvl1);
    cudaFree(tablelvl1);
    cout<<"PASS"<<endl;
}