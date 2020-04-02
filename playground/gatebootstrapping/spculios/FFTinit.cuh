#pragma once
#include<cuparams.hpp>
#include<cmath>

namespace SPCULIOS{
    __device__ double twistlvl1[TFHEpp::DEF_N],tablelvl1[TFHEpp::DEF_N];

    template<uint32_t N>
    inline std::array<double,N> TwistGen(){
        std::array<double,N> twist;
        for(uint32_t i = 0;i<N/2;i++){
            twist[i] = std::cos(i*M_PI/N);
            twist[i+N/2] = std::sin(i*M_PI/N);
        }
        return twist;
    }

    template<uint32_t N>
    inline std::array<double,2*N> TableGen(){
        std::array<double,2*N> table;
        for(uint32_t i = 0;i<N;i++){
            table[i] = std::cos(2*i*M_PI/N);
            table[i+N] = std::sin(2*i*M_PI/N);
        }
        return table;
    }


    void FFTinit(){
        const std::array<double,TFHEpp::DEF_N> h_twistlvl1 = TwistGen<TFHEpp::DEF_N>();
        const std::array<double,TFHEpp::DEF_N> h_tablelvl1 = TableGen<TFHEpp::DEF_N/2>();
        cudaMemcpyToSymbol(twistlvl1, h_twistlvl1.data(), sizeof(h_twistlvl1));
        cudaMemcpyToSymbol(tablelvl1, h_tablelvl1.data(), sizeof(h_tablelvl1));
    }

}