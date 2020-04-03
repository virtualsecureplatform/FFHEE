#pragma once
#include<params.hpp>
#include<cloudkey.hpp>
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

__device__ inline void PolynomialMulByXaiMinusOneAndDecompositionFFTlvl1(cuDecomposedPolynomialInFDlvl1 decvecfft, const cuPolynomiallvl1 poly, const uint32_t a, const double* twist, const double* table){
    const unsigned int tidy = threadIdx.y;
    static constexpr uint32_t offset = offsetgenlvl1();
    PolynomialMulByXaiMinusOneAndDecomposition<uint32_t, TFHEpp::DEF_Nbit, TFHEpp::DEF_N, TFHEpp::DEF_l, TFHEpp::DEF_Bgbit, offset>(decvecfft, poly, a);
    TwistIFFTinPlacelvl1(decvecfft[tidy],twist,table);
}


__device__ void BlindRotateFFTlvl1(cuTRLWElvl1 trlwe, const cuTRGSWFFTlvl1 trgswfft, const uint32_t a, const double* twist, const double* table, uint8_t* const smem){
    cuPolynomialInFDlvl1 *decvecfft = (double(*)[TFHEpp::DEF_N])&smem[0];
    cuPolynomialInFDlvl1 *restrlwefft = (double(*)[TFHEpp::DEF_N])&decvecfft[TFHEpp::DEF_l];

    PolynomialMulByXaiMinusOneAndDecompositionFFTlvl1(decvecfft, trlwe[0], a, twist, table);
    MulInFD<TFHEpp::DEF_N>(restrlwefft[0], decvecfft[0], trgswfft[0][0]);
    MulInFD<TFHEpp::DEF_N>(restrlwefft[1], decvecfft[0], trgswfft[0][1]);
    #pragma unroll
    for (int i = 1; i < TFHEpp::DEF_l; i++) {
        FMAInFD<TFHEpp::DEF_N>(restrlwefft[0], decvecfft[i], trgswfft[i][0]);
        FMAInFD<TFHEpp::DEF_N>(restrlwefft[1], decvecfft[i], trgswfft[i][1]);
    }

    PolynomialMulByXaiMinusOneAndDecompositionFFTlvl1(decvecfft, trlwe[1], a, twist, table);

    #pragma unroll
    for (int i = 0; i < TFHEpp::DEF_l; i++) {
        FMAInFD<TFHEpp::DEF_N>(restrlwefft[0], decvecfft[i], trgswfft[i+TFHEpp::DEF_l][0]);
        FMAInFD<TFHEpp::DEF_N>(restrlwefft[1], decvecfft[i], trgswfft[i+TFHEpp::DEF_l][1]);
    }

    cuPolynomiallvl1 *buff = (uint32_t (*)[TFHEpp::DEF_N])&decvecfft[0];
    TwistFFTlvl1(buff[threadIdx.y], restrlwefft[threadIdx.y],twist,table);

    const unsigned int tid = blockDim.x*threadIdx.y+threadIdx.x;
    const unsigned int bdim = blockDim.x*blockDim.y;
    #pragma unroll
    for(int i = tid;i<TFHEpp::DEF_N;i+=bdim){
        trlwe[0][i] += buff[0][i];
        trlwe[1][i] += buff[1][i];
    }
    __syncthreads();
}

template <uint32_t Mbit = TFHEpp::DEF_Nbit+1>
__device__ inline uint32_t modSwitchFromTorus32(uint32_t phase)
{
    return (phase+(1U<<(31-Mbit)))>>(32-Mbit);
}

template <typename T = uint32_t, uint32_t Nbit = TFHEpp::DEF_Nbit, uint32_t N = TFHEpp::DEF_N>
__device__ inline void RotatedTestVector(T trlwe[2][N], const T bar, const T mu)
{
    const unsigned int tid = blockDim.x*threadIdx.y+threadIdx.x;
    const unsigned int bdim = blockDim.x*blockDim.y;
    
    constexpr T cmpmask = N-1;

    T cmp, neg, pos;
#pragma unroll
    for (T i = tid; i < N; i += bdim) {
        trlwe[0][i] = 0;  // part a
        if (bar == 2 * N)
            trlwe[1][i] = mu;
        else {
            cmp = (T)(i < (bar & cmpmask));
            neg = -(cmp ^ (bar >> Nbit));
            pos = -((1 - cmp) ^ (bar >> Nbit));
            trlwe[1][i] = (mu & pos) + ((-mu) & neg);  // part b
        }
    }
    __syncthreads();
}

__device__ cuBootStrappingKeyFFTlvl01 d_bkfftlvl01;
__device__ cuKeySwitchingKey d_ksk;
__device__ cuTLWElvl0 d_tlwe[NUMBER_OF_STREAM],d_res[NUMBER_OF_STREAM];

__device__ void __GateBootstrappingTLWE2TRLWEFFTlvl01__(cuTRLWElvl1 acc, const cuTLWElvl0 tlwe, uint8_t* const smem){
    double* twist = (double*)&smem[0];
    double* table = (double*)&twist[TFHEpp::DEF_N];
    const unsigned int tid = blockDim.x*threadIdx.y+threadIdx.x;
    const unsigned int bdim = blockDim.x*blockDim.y;

    #pragma unroll
    for(int i = tid;i<TFHEpp::DEF_N;i+=bdim){
        twist[i] = SPCULIOS::twistlvl1[i];
        table[i] = SPCULIOS::tablelvl1[i];
    }
    __syncthreads();
    uint32_t bara = 2 * TFHEpp::DEF_N - modSwitchFromTorus32<TFHEpp::DEF_Nbit+1>(tlwe[TFHEpp::DEF_n]);
    RotatedTestVector<uint32_t, TFHEpp::DEF_Nbit, TFHEpp::DEF_N>(acc, bara, DEF_MU);
    #pragma unroll
    for(int i = 0; i<TFHEpp::DEF_n;i++){
        bara = modSwitchFromTorus32(tlwe[i]);
        BlindRotateFFTlvl1(acc,d_bkfftlvl01[i],bara,twist,table,(uint8_t*)&table[TFHEpp::DEF_N]);
    }
}

__device__ inline void SampleExtractIndexZeroAndIdentityKeySwitchlvl10(cuTLWElvl0 tlwe, cuTRLWElvl1 trlwe)
{
    const unsigned int tid = blockDim.x*threadIdx.y+threadIdx.x;
    const unsigned int bdim = blockDim.x*blockDim.y;

    constexpr uint32_t prec_offset = 1U << (32 - (1 + TFHEpp::DEF_basebit * TFHEpp::DEF_t));
    constexpr uint32_t mask = (1U << TFHEpp::DEF_basebit) - 1;

#pragma unroll
    for (int i = tid; i <= TFHEpp::DEF_n; i += bdim) {
        uint32_t tmp;
        uint32_t res = 0;
        uint32_t val = 0;
        if (i == TFHEpp::DEF_n) res = trlwe[1][0];
#pragma unroll 0
        for (int j = 0; j < TFHEpp::DEF_N; j++) {
            if (j == 0)
                tmp = trlwe[0][0];
            else
                tmp = -trlwe[0][TFHEpp::DEF_N - j];
            tmp += prec_offset;
            #pragma unroll
            for (int k = 0; k < TFHEpp::DEF_t; k++) {
                val = (tmp >> (32 - (k + 1) * TFHEpp::DEF_basebit)) & mask;
                if (val != 0)
                    res -= d_ksk[j][k][val-1][i];
            }
        }
        tlwe[i] = res;
    }
    __threadfence();
}

__global__ void __GateBootstrapping__(cuTLWElvl0* res, cuTLWElvl0* tlwe){
    extern __shared__ uint8_t smem[];
    cuPolynomiallvl1 *trlwe = (uint32_t (*)[TFHEpp::DEF_N])(uint32_t (*)[TFHEpp::DEF_N])&smem[0];
    __GateBootstrappingTLWE2TRLWEFFTlvl01__(trlwe,*tlwe,(uint8_t*)&trlwe[2]);
    SampleExtractIndexZeroAndIdentityKeySwitchlvl10(*res,trlwe);
}

void FFHEEinit(TFHEpp::GateKey &gk){
    FFTinit();
    cudaMemcpyToSymbol(d_bkfftlvl01,gk.bkfftlvl01.data(),sizeof(gk.bkfftlvl01));
    cudaMemcpyToSymbol(d_ksk,gk.ksk.data(),sizeof(gk.ksk));
    cudaFuncSetAttribute(__GateBootstrapping__,cudaFuncAttributePreferredSharedMemoryCarveout,80);
    cudaFuncSetAttribute(__GateBootstrapping__,cudaFuncAttributeMaxDynamicSharedMemorySize,64*1024);
}

void GateBootstrapping(TFHEpp::TLWElvl0 &res, const TFHEpp::TLWElvl0 &tlwe,
                                 const cudaStream_t &st, uint32_t index){
    cuTLWElvl0 *pt_tlwe,*pt_res;
    cudaGetSymbolAddress((void**)&pt_tlwe,d_tlwe);
    cudaGetSymbolAddress((void**)&pt_res,d_res);
    cudaMemcpyAsync(pt_tlwe+index,tlwe.data(),sizeof(tlwe),cudaMemcpyHostToDevice,st);
    __GateBootstrapping__<<<1,dim3(TFHEpp::DEF_N/16,TFHEpp::DEF_l,1),64*1024,st>>>(&pt_res[index],&pt_tlwe[index]);
    cudaMemcpyAsync(res.data(),pt_res+index,sizeof(res),cudaMemcpyDeviceToHost, st);
}
}