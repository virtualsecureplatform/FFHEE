#pragma once
#include<params.hpp>
#include<spqlios++.hpp>

namespace SPCULIOS{
    __device__ double twistlvl1[TFHEpp::DEF_N],tablelvl1[TFHEpp::DEF_N];

    void FFTinit(){
        const std::array<double,TFHEpp::DEF_N> h_twistlvl1 = SPQLIOSpp::TwistGen<TFHEpp::DEF_N>();
        const std::array<double,TFHEpp::DEF_N> h_tablelvl1 = SPQLIOSpp::TableGen<TFHEpp::DEF_N/2>();
        cudaMemcpyToSymbol(twistlvl1, h_twistlvl1.data(), sizeof(h_twistlvl1));
        cudaMemcpyToSymbol(tablelvl1, h_tablelvl1.data(), sizeof(h_tablelvl1));
    }

}