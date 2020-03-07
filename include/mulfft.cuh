#pragma once

#include<utility>
#include <type_traits>
#include <stdint.h>

#include "params.hpp"

namespace FFHEE{
    using namespace std;

    __constant__ static double twistlvl1[FFHEE_DEF_N];
    __constant__ static double tablelvl1[2*FFHEE_DEF_N];
    __constant__ static double twistlvl2[FFHEE_DEF_nbar];
    __constant__ static double,tablelvl2[2*FFHEE_DEF_nbar];

    template<uint32_t N>
    static inline array<double,N> TwistGen(){
        array<double,N> twist;
        for(uint32_t i = 0;i<N/2;i++){
            twist[i] = cos(i*M_PI/N);
            twist[i+N/2] = sin(i*M_PI/N);
        }
        return twist;
    }

    template<uint32_t N>
    static inline array<double,2*N> TableGen(){
        array<double,2*N> table;
        for(uint32_t i = 0;i<N;i++){
            table[i] = cos(2*i*M_PI/N);
            table[i+N] = sin(2*i*M_PI/N);
        }
        return table;
    }

    void FFTInit(){
        cudaMemcpyToSymbol(twistlvl1,TwistGen<FFHEE_DEF_N>().data(),sizeof(double)*FFHEE_DEF_N);
        cudaMemcpyToSymbol(tablelvl1,TableGen<FFHEE_DEF_N>().data(),sizeof(double)*2*FFHEE_DEF_N);
        cudaMemcpyToSymbol(twistlvl2,TwistGen<FFHEE_DEF_nbar>().data(),sizeof(double)*FFHEE_DEF_nbar);
        cudaMemcpyToSymbol(tablelvl2,TableGen<FFHEE_DEF_nbar>().data(),sizeof(double)*2*FFHEE_DEF_nbar);
    }

    template<typename T = uint32_t,uint32_t N = FFHEE_DEF_N>
    __device__ inline void TwistMulInvert(double* res, const T* a, const double* twist){
        const unsigned int tid = threadIdx.x;
        for (int i = 0; i < N / 2; i+=tid) {
            const double are = static_cast<double>(static_cast<typename make_signed<T>::type>(a[i]));
            const double aim = static_cast<double>(static_cast<typename make_signed<T>::type>(a[i+N/2]));
            const double aimbim = aim * twist[i + N / 2];
            const double arebim = are * twist[i + N / 2];
            res[i] = are * twist[i] - aimbim;
            res[i + N / 2] = aim * twist[i] + arebim;
        }
        __syncthreads();
    }

    template<uint32_t N = FFHEE_DEF_N>
    __device__ inline void TwistMulDirectlvl1(uint32_t* res, const double* a, const double* twist){
        const unsigned int tid = threadIdx.x;
        for (int i = 0; i < N / 2; i+=tid) {
            const double aimbim = a[i + N / 2] * -twist[i + N / 2];
            const double arebim = a[i] * -twist[i + N / 2];
            res[i] = static_cast<int64_t>((a[i] * twist[i] - aimbim)*(2.0/N));
            res[i + N / 2] = static_cast<int64_t>((a[i + N / 2] * twist[i] + arebim)*(2.0/N));
        }
        __syncthreads();
    }

    template<uint32_t N = FFHEE_DEF_nbar>
    __device__ inline void TwistMulDirectlvl2(uint64_t*res, const double* a, const double* twist){
        const unsigned int tid = threadIdx.x;

        constexpr uint64_t valmask0 = 0x000FFFFFFFFFFFFFul;
        constexpr uint64_t valmask1 = 0x0010000000000000ul;
        constexpr uint16_t expmask0 = 0x07FFu;
        for (int i = 0; i < N / 2; i+=tid) {
            const double aimbim = a[i + N / 2] * -twist[i + N / 2];
            const double arebim = a[i] * -twist[i + N / 2];
            const double resdoublere = (a[i] * twist[i] - aimbim)*(2.0/N);
            const double resdoubleim = (a[i + N / 2] * twist[i] + arebim)*(2.0/N);
            const uint64_t resre = reinterpret_cast<const uint64_t&>(resdoublere);
            const uint64_t resim = reinterpret_cast<const uint64_t&>(resdoubleim);

            uint64_t val = (resre&valmask0)|valmask1; //mantissa on 53 bits
            uint16_t expo = (resre>>52)&expmask0; //exponent 11 bits
            // 1023 -> 52th pos -> 0th pos
            // 1075 -> 52th pos -> 52th pos
            int16_t trans = expo-1075;
            uint64_t val2 = trans>0?(val<<trans):(val>>-trans);
            res[i] = (resre>>63)?-val2:val2;

            val = (resim&valmask0)|valmask1; //mantissa on 53 bits
            expo = (resim>>52)&expmask0; //exponent 11 bits
            // 1023 -> 52th pos -> 0th pos
            // 1075 -> 52th pos -> 52th pos
            trans = expo-1075;
            val2 = trans>0?(val<<trans):(val>>-trans);
            res[i + N / 2] = (resim>>63)?-val2:val2;
            
        }
        __syncthreads();
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

    template <uint32_t Nbit = FFHEE_DEF_Nbit, uint32_t stride, bool isinvert =true>
    __device__ inline void TwiddleMul(double* const a, const array<double,1<<(Nbit+1)> &table, const uint32_t step, const unsigned int i){
        constexpr uint32_t N = 1<<Nbit;

        double& are = *(a);
        double& aim = *(a+N);
        const double bre = table[stride*(1<<step)*i];
        const double bim = isinvert?table[stride*(1<<step)*i + N]:-table[stride*(1<<step)*i + N];

        const double aimbim = aim * bim;
        const double arebim = are * bim;
        are = are * bre - aimbim;
        aim = aim * bre + arebim;
    }

    template <uint32_t Nbit = FFHEE_DEF_Nbit-1, bool isinvert =true>
    __device__ inline void Radix4TwiddleMul(double* const a, int i){
        constexpr uint32_t N = 1<<Nbit;

        double* const are = a;
        double* const aim = a+N;
        swap(are[i],aim[i]);
        if constexpr(isinvert){
            are[i]*=-1;
        }
        else{
            aim[i]*=-1;
        }
    }

    template <uint32_t Nbit = FFHEE_DEF_Nbit-1, bool isinvert =true>
    __device__ inline void Radix8TwiddleMulStrideOne(double* const a,int i){
        constexpr uint32_t N = 1<<Nbit;

        double* const are = a;
        double* const aim = a+N;
        const double _1sroot2 = 1/sqrt(2);
        const double aimbim = isinvert?aim[i]:-aim[i];
        const double arebim = isinvert?are[i]:-are[i];
        are[i] = _1sroot2*(are[i] - aimbim);
        aim[i] = _1sroot2*(aim[i] + arebim);
    }

    template <uint32_t Nbit = FFHEE_DEF_Nbit-1, bool isinvert =true>
    __device__ inline void Radix8TwiddleMulStrideThree(double* const a, int i){
        constexpr uint32_t N = 1<<Nbit;

        double* const are = a;
        double* const aim = a+N;
        const double _1sroot2 = 1/sqrt(2);
        const double aimbim = isinvert?aim[i]:-aim[i];
        const double arebim = isinvert?are[i]:-are[i];
        are[i] = _1sroot2*(-are[i] - aimbim);
        aim[i] = _1sroot2*(-aim[i] + arebim);
    }

    template<uint32_t Nbit = FFHEE_DEF_Nbit-1>
    __device__ inline void IFFT(array<double,1<<(Nbit+1)>& res, const array<double,1<<(Nbit+1)> &table){
        const unsigned int tid = threadIdx.x;
        
        constexpr uint32_t N = 1<<Nbit;

        for(int step = 0;step<Nbit;step++){
            const unsigned int size = 1<<(Nbit-step);
            const unsigned int index = tid>>step;
            double* const res0 = &res[(tid & ((1<<step)-1))*size+index];
            double* const res1 = res0+size/2;
            ButterflyAdd<N>(res0,res1);
            TwiddleMul<Nbit,1,false>(res1,table,index);
            __syncthreads();
        }
    }

    __device__ inline void TwistIFFTlvl1(double* res, const uint32_t* a){
        TwistMulInvert<uint32_t,FFHEE_DEF_N>(res,a,twistlvl1);
        IFFT<FFHEE_DEF_Nbit-1>(res,tablelvl1);
    }

    __device__ inline void TwistIFFTlvl2(double* res, const uint64_t* a){
        TwistMulInvert<uint64_t,FFHEE_DEF_nbar>(res,a,twistlvl2);
        IFFT<FFHEE_DEF_nbarbit-1>(res,tablelvl2);
    }

    template<uint32_t Nbit = FFHEE_DEF_Nbit-1>
    __device__ inline void FFT(double* res, const double* table){
        const unsigned int tid = threadIdx.x;
        
        constexpr uint32_t N = 1<<Nbit;
        for(int step = 0;step<Nbit;step++){
            const unsigned int size = 1<<(step+1);
            const unsigned int index = tid>>(Nbit-1+step);
            double* const res0 = &res[(tid & ((1<<(Nbit-1+step))-1))*size+index];
            double* const res1 = res0+size/2;
            TwiddleMul<Nbit,1,true>(res1,table,index);
            ButterflyAdd<<<1, cuFHE_DEF_N / 2, 0, st>>><N>(res0,res1);
            __syncthreads();
        }
        
    }

    __device__ inline void TwistFFTlvl1(uint32_t* res, double* a){
        FFT<FFHEE_DEF_Nbit-1>(a, tablelvl1);
        TwistMulDirectlvl1<FFHEE_DEF_N>(res,a,twistlvl1);
    }

    __device__ inline void TwistFFTlvl2(uint64_t* res, double* &a){
        FFT<FFHEE_DEF_nbarbit-1>(a.data(),tablelvl2);
        TwistMulDirectlvl2<FFHEE_DEF_nbar>(res,a,twistlvl2);
    }

    __global__ void FFTlvl1Test(Polynomiallvl1 res,const Polynomiallvl1 a){
        uint32_t* deva;
        cudaMalloc((void**)&deva,sizeof(a));
        double* temp;
        cudaMalloc((void**)&temp,sizeof(double)*DEF_N);

        cudaMemcpy(a.data(),deva,sizeof(a),cudaMemcpyDeviceToHost);

        TwistIFFTlvl1<<<1,64>>>(temp,deva);
        TwistFFTlvl1<<<1,64>>>(deva,temp);

        cudaMemcpy(deva,res.data(),sizeof(a),cudaMemcpyHostToDevice);

        cudaFree(deva);
        cudaFree(temo);
    }

    __device__ inline void PolyMullvl1(Polynomiallvl1 &res, const Polynomiallvl1 &a,
                        const Polynomiallvl1 &b)
    {
        PolynomialInFDlvl1 ffta;
        TwistIFFTlvl1(ffta, a);
        PolynomialInFDlvl1 fftb;
        TwistIFFTlvl1(fftb, b);
        MulInFD<DEF_N>(ffta, ffta, fftb);
        TwistFFTlvl1(res, ffta);
    }
}