#pragma once
#include<params.hpp>
#include"cuparams.hpp"
#include"trgsw.cuh"

namespace FFHEE{
template <typename T = uint32_t, uint32_t Nbit = TFHEpp::DEF_Nbit, uint32_t N = TFHEpp::DEF_N, uint32_t l = TFHEpp::DEF_l,
          uint32_t Bgbit = TFHEpp::DEF_Bgbit, T offset>
__device__ inline void trlweMulByXaiMinusOneAndDecomposition(double decvec[2*l][N],
                                       const T trlwe[2][N], const T a)
{
    const unsigned int tid = blockDim.x*threadIdx.y+threadIdx.x;
    const unsigned int bdim = blockDim.x*blockDim.y;

    constexpr T mask = static_cast<T>((1UL << Bgbit) - 1);
    constexpr T halfBg = (1UL << (Bgbit - 1));
    constexpr T cmpmask = N-1;

    T temp;
#pragma unroll
    for (int i = tid; i < N; i += bdim) {
        const T cmp = (uint32_t)(i < (a & cmpmask));
        const T neg = -(cmp ^ (a >> Nbit));
        const T pos = -((1 - cmp) ^ (a >> Nbit));
#pragma unroll
        for (int j = 0; j < 2; j++) {
            temp = trlwe[j][(i - a) & cmpmask];
            temp = (temp & pos) + ((-temp) & neg);
            temp -= trlwe[j][i];
            // decomp temp
            temp += offset;
            #pragma unroll
            for(int k =  0; k<l; k++)
            decvec[l*j+k][i] = static_cast<double>(static_cast<typename std::make_signed<T>::type>(
                ((temp >> (32 - (k+1)*Bgbit)) & mask) - halfBg));
        }
    }
    __syncthreads();
}

__device__ inline void trlweMulByXaiMinusOneAndDecompositionFFTlvl1(cuDecomposedTRLWEInFDlvl1 decvecfft, const cuTRLWElvl1 trlwe, const uint32_t a){
    const unsigned int tidy = threadIdx.y;
    static constexpr uint32_t offset = offsetgenlvl1();
    trlweMulByXaiMinusOneAndDecomposition<uint32_t, TFHEpp::DEF_Nbit, TFHEpp::DEF_N, TFHEpp::DEF_l, TFHEpp::DEF_Bgbit, offset>(decvecfft, trlwe, a);
    for (int i = 0; i < TFHEpp::DEF_l; i++) TwistIFFTinPlacelvl1(decvecfft[i+tidy*TFHEpp::DEF_l]);
}

__device__ cuTRGSWFFTlvl1 d_trgswfft;
__device__ cuTRLWElvl1 d_res,d_trlwe;
__device__ uint32_t d_a;

__global__ void __BlindRotateFFTlvl1__(){
    __shared__ cuDecomposedTRLWEInFDlvl1 decvecfft;
    trlweMulByXaiMinusOneAndDecompositionFFTlvl1(decvecfft, d_trlwe, d_a);
    __shared__ cuTRLWEInFDlvl1 restrlwefft;
    MulInFD<TFHEpp::DEF_N>(restrlwefft[0], decvecfft[0], d_trgswfft[0][0]);
    MulInFD<TFHEpp::DEF_N>(restrlwefft[1], decvecfft[0], d_trgswfft[0][1]);
    for (int i = 1; i < 2 * TFHEpp::DEF_l; i++) {
        FMAInFD<TFHEpp::DEF_N>(restrlwefft[0], decvecfft[i], d_trgswfft[i][0]);
        FMAInFD<TFHEpp::DEF_N>(restrlwefft[1], decvecfft[i], d_trgswfft[i][1]);
    }
    TwistFFTlvl1(d_res[threadIdx.y], restrlwefft[threadIdx.y]);

    const unsigned int tid = blockDim.x*threadIdx.y+threadIdx.x;
    const unsigned int bdim = blockDim.x*blockDim.y;
    for(int i = tid;i<TFHEpp::DEF_N;i+=bdim){
        d_res[0][i] += d_trlwe[0][i];
        d_res[1][i] += d_trlwe[1][i];
    }
    __threadfence();
}

void BlindRotateFFTlvl1(TFHEpp::TRLWElvl1 &res, const TFHEpp::TRLWElvl1 &trlwe,
                                 const TFHEpp::TRGSWFFTlvl1 &trgswfft, const uint32_t a){
    cudaMemcpyToSymbolAsync(d_trlwe,trlwe.data(),sizeof(trlwe));
    cudaMemcpyToSymbolAsync(d_trgswfft,trgswfft.data(),sizeof(trgswfft));
    cudaMemcpyToSymbolAsync(d_a,&a,sizeof(a));
    __BlindRotateFFTlvl1__<<<1,dim3(64,2,1)>>>();
    cudaMemcpyFromSymbolAsync(res.data(),d_res,sizeof(res));
}
}