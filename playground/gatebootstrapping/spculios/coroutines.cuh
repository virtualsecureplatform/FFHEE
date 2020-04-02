#pragma once
#include <math.h>
#include <cuparams.hpp> 

namespace SPCULIOS{

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

    template <uint32_t Nbit = TFHEpp::DEF_Nbit-1, uint32_t stride, bool isinvert =true>
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

    template <uint32_t Nbit = TFHEpp::DEF_Nbit-1, bool isinvert =true>
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

    template <uint32_t Nbit = TFHEpp::DEF_Nbit-1, bool isinvert =true>
    __device__ inline void Radix8TwiddleMulStrideOne(double* const a){
        constexpr uint32_t N = 1<<Nbit;

        double& are = *a;
        double& aim = *(a+N);
        const double aimbim = isinvert?aim:-aim;
        const double arebim = isinvert?are:-are;
        are = M_SQRT1_2 *(are - aimbim);
        aim = M_SQRT1_2 *(aim + arebim);
    }

    template <uint32_t Nbit = TFHEpp::DEF_Nbit-1, bool isinvert =true>
    __device__ inline void Radix8TwiddleMulStrideThree(double* const a){
        constexpr uint32_t N = 1<<Nbit;

        double& are = *a;
        double& aim = *(a+N);
        const double aimbim = isinvert?aim:-aim;
        const double arebim = isinvert?are:-are;
        are = M_SQRT1_2 *(-are - aimbim);
        aim = M_SQRT1_2 *(-aim + arebim);
    }
}