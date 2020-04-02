#pragma once
#include<params.hpp>
#include"cuparams.hpp"
#include"trgsw.cuh"

namespace FFHEE{
template <typename T = uint32_t, uint32_t Nbit = TFHEpp::DEF_Nbit, uint32_t N = TFHEpp::DEF_N, uint32_t l = TFHEpp::DEF_l,
          uint32_t Bgbit = TFHEpp::DEF_Bgbit, T offset>
__device__ inline void PolynomialMulByXaiMinusOneAndDecomposition(double decvec[l][N],
                                       const T poly[N], const T a)
{
    const unsigned int tid = blockDim.x*threadIdx.y+threadIdx.x;
    const unsigned int bdim = blockDim.x*blockDim.y;

    constexpr T mask = static_cast<T>((1UL << Bgbit) - 1);
    constexpr T halfBg = (1UL << (Bgbit - 1));
    constexpr T cmpmask = N-1;

    T temp;
#pragma unroll
    for (T i = tid; i < N; i += bdim) {
        const T cmp = (T)(i < (a & cmpmask));
        const T neg = -(cmp ^ (a >> Nbit));
        const T pos = -((1 - cmp) ^ (a >> Nbit));
#pragma unroll
        temp = poly[(i - a) & cmpmask];
        temp = (temp & pos) + ((-temp) & neg);
        temp -= poly[i];
        // decomp temp
        temp += offset;
        #pragma unroll
        for(int k =  0; k<l; k++)
        decvec[k][i] = static_cast<double>(static_cast<typename std::make_signed<T>::type>(
                ((temp >> (32 - (k+1)*Bgbit)) & mask) - halfBg));
    }
    __syncthreads();
}

__device__ inline void PolynomialMulByXaiMinusOneAndDecompositionFFTlvl1(cuDecomposedPolynomialInFDlvl1 decvecfft, const cuPolynomiallvl1 poly, const uint32_t a){
    const unsigned int tidy = threadIdx.y;
    static constexpr uint32_t offset = offsetgenlvl1();
    PolynomialMulByXaiMinusOneAndDecomposition<uint32_t, TFHEpp::DEF_Nbit, TFHEpp::DEF_N, TFHEpp::DEF_l, TFHEpp::DEF_Bgbit, offset>(decvecfft, poly, a);
    #pragma unroll
    TwistIFFTinPlacelvl1(decvecfft[tidy]);
}

__device__ cuTRGSWFFTlvl1 d_trgswfft;
__device__ cuTRLWElvl1 d_trlwe;
__device__ uint32_t d_a;

__global__ void __BlindRotateFFTlvl1__(){
    extern __shared__ uint8_t smem[];
    cuPolynomialInFDlvl1 *decvecfft = (double(*)[TFHEpp::DEF_N])&smem[0];
    cuPolynomialInFDlvl1 *restrlwefft = (double(*)[TFHEpp::DEF_N])&decvecfft[TFHEpp::DEF_l];

    PolynomialMulByXaiMinusOneAndDecompositionFFTlvl1(decvecfft, d_trlwe[0], d_a);
    MulInFD<TFHEpp::DEF_N>(restrlwefft[0], decvecfft[0], d_trgswfft[0][0]);
    MulInFD<TFHEpp::DEF_N>(restrlwefft[1], decvecfft[0], d_trgswfft[0][1]);
    #pragma unroll
    for (int i = 1; i < TFHEpp::DEF_l; i++) {
        FMAInFD<TFHEpp::DEF_N>(restrlwefft[0], decvecfft[i], d_trgswfft[i][0]);
        FMAInFD<TFHEpp::DEF_N>(restrlwefft[1], decvecfft[i], d_trgswfft[i][1]);
    }

    PolynomialMulByXaiMinusOneAndDecompositionFFTlvl1(decvecfft, d_trlwe[1], d_a);

    for (int i = 0; i < TFHEpp::DEF_l; i++) {
        FMAInFD<TFHEpp::DEF_N>(restrlwefft[0], decvecfft[i], d_trgswfft[i+TFHEpp::DEF_l][0]);
        FMAInFD<TFHEpp::DEF_N>(restrlwefft[1], decvecfft[i], d_trgswfft[i+TFHEpp::DEF_l][1]);
    }

    cuPolynomiallvl1 *buff = (uint32_t (*)[TFHEpp::DEF_N])&decvecfft[0];
    TwistFFTlvl1(buff[threadIdx.y], restrlwefft[threadIdx.y]);

    const unsigned int tid = blockDim.x*threadIdx.y+threadIdx.x;
    const unsigned int bdim = blockDim.x*blockDim.y;
    for(int i = tid;i<TFHEpp::DEF_N;i+=bdim){
        d_trlwe[0][i] += buff[0][i];
        d_trlwe[1][i] += buff[1][i];
    }
    __threadfence();
}

void BlindRotateFFTlvl1(TFHEpp::TRLWElvl1 &res, const TFHEpp::TRLWElvl1 &trlwe,
                                 const TFHEpp::TRGSWFFTlvl1 &trgswfft, const uint32_t a){
    cudaMemcpyToSymbolAsync(d_trlwe,trlwe.data(),sizeof(trlwe));
    cudaMemcpyToSymbolAsync(d_trgswfft,trgswfft.data(),sizeof(trgswfft));
    cudaMemcpyToSymbolAsync(d_a,&a,sizeof(a));
    __BlindRotateFFTlvl1__<<<1,dim3(TFHEpp::DEF_N/16,TFHEpp::DEF_l,1),32*1024>>>();
    cudaMemcpyFromSymbolAsync(res.data(),d_trlwe,sizeof(res));
}
}