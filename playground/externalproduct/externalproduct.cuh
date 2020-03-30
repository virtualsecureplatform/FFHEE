#pragma once
#include<params.hpp>
#include"mulfft/mulfft.cuh"
#include"cuparams.hpp"

#include <limits>

namespace FFHEE{
using namespace std;
using namespace SPCULIOS;

template <typename T = uint32_t, uint32_t N = TFHEpp::DEF_N, uint32_t l = TFHEpp::DEF_l,
          uint32_t Bgbit = TFHEpp::DEF_Bgbit, T offset>
__device__ inline void Decomposition(double decvec[2*l][N], const T trlwe[2][N]){
    const unsigned int tid = blockDim.x*threadIdx.y+threadIdx.x;
    const unsigned int bdim = blockDim.x*blockDim.y;

    constexpr T mask = static_cast<T>((1UL << Bgbit) - 1);
    constexpr T halfBg = (1UL << (Bgbit - 1));

    for (int j = tid; j < N; j+=bdim) {
        T temp0 = trlwe[0][j] + offset;
        T temp1 = trlwe[1][j] + offset;
        for (int i = 0; i < l; i++)
            decvec[i][j] = static_cast<double>(static_cast<typename std::make_signed<T>::type>(((temp0 >>
                             (numeric_limits<T>::digits - (i + 1) * Bgbit)) &
                            mask) -
                           halfBg));
        for (int i = 0; i < l; i++)
            decvec[i + l][j] = static_cast<double>(static_cast<typename std::make_signed<T>::type>(((temp1 >> (numeric_limits<T>::digits -
                                                  (i + 1) * Bgbit)) &
                                mask) -
                               halfBg));
    }
    __syncthreads();
}

__device__ constexpr uint32_t offsetgenlvl1()
{
    uint32_t offset = 0;
    for (int i = 1; i <= TFHEpp::DEF_l; i++)
        offset += TFHEpp::DEF_Bg / 2 * (1U << (32 - i * TFHEpp::DEF_Bgbit));
    return offset;
}

__device__ inline void DecompositionFFTlvl1(cuDecomposedTRLWEInFDlvl1 decvecfft,
                                 const cuTRLWElvl1 trlwe)
{
    const unsigned int tidy = threadIdx.y;
    static constexpr uint32_t offset = offsetgenlvl1();
    Decomposition<uint32_t, TFHEpp::DEF_N, TFHEpp::DEF_l, TFHEpp::DEF_Bgbit, offset>(decvecfft, trlwe);
    for (int i = 0; i < TFHEpp::DEF_l; i++) TwistIFFTinPlacelvl1(decvecfft[i+tidy*TFHEpp::DEF_l]);
    __syncthreads();
}

__device__ cuTRGSWFFTlvl1 d_trgswfft;
__device__ cuTRLWElvl1 d_res,d_trlwe;

__global__ void __trgswfftExternalProductlvl1__(){
    __shared__ cuDecomposedTRLWEInFDlvl1 decvecfft;
    DecompositionFFTlvl1(decvecfft, d_trlwe);
    __shared__ cuTRLWEInFDlvl1 restrlwefft;
    MulInFD<TFHEpp::DEF_N>(restrlwefft[0], decvecfft[0], d_trgswfft[0][0]);
    MulInFD<TFHEpp::DEF_N>(restrlwefft[1], decvecfft[0], d_trgswfft[0][1]);
    for (int i = 1; i < 2 * TFHEpp::DEF_l; i++) {
        FMAInFD<TFHEpp::DEF_N>(restrlwefft[0], decvecfft[i], d_trgswfft[i][0]);
        FMAInFD<TFHEpp::DEF_N>(restrlwefft[1], decvecfft[i], d_trgswfft[i][1]);
    }
    TwistFFTlvl1(d_res[threadIdx.y], restrlwefft[threadIdx.y]);
}

void trgswfftExternalProductlvl1(TFHEpp::TRLWElvl1 &res, const TFHEpp::TRLWElvl1 &trlwe,
                                 const TFHEpp::TRGSWFFTlvl1 &trgswfft){
        cudaMemcpyToSymbolAsync(d_trlwe,trlwe.data(),sizeof(trlwe));
        cudaMemcpyToSymbolAsync(d_trgswfft,trgswfft.data(),sizeof(trgswfft));
        __trgswfftExternalProductlvl1__<<<1,dim3(64,2,1)>>>();
        cudaMemcpyFromSymbolAsync(res.data(),d_res,sizeof(res));
    }
}