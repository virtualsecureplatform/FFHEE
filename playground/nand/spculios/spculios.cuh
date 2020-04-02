#pragma once
#include<params.hpp>
#include"FFTinit.cuh"
#include"coroutines.cuh"
#include"TwistFFT.cuh"
#include"TwistIFFT.cuh"

namespace SPCULIOS{
    using namespace FFHEE;
    
    template<uint32_t N = TFHEpp::DEF_N>
    __device__ inline void MulInFD(double* res, const double* a, const double* b){
        const unsigned int tid = blockDim.x*threadIdx.y+threadIdx.x;
        const unsigned int bdim = blockDim.x*blockDim.y;

        for (int i = tid; i < N / 2; i+=bdim) {
            const double aimbim = a[i + N / 2] * b[i + N / 2];
            const double arebim = a[i] * b[i + N / 2];
            res[i] = a[i] * b[i] - aimbim;
            res[i + N / 2] = a[i + N / 2] * b[i] + arebim;
        }
        __syncthreads();
    }

    template <uint32_t N>
    __device__ inline void FMAInFD(double* res, const double* a,
                        const double* b)
    {
        const unsigned int tid = blockDim.x*threadIdx.y+threadIdx.x;
        const unsigned int bdim = blockDim.x*blockDim.y;
        
        for (int i = tid; i < N / 2; i+=bdim) {
            res[i] = a[i + N / 2] * b[i + N / 2] - res[i];
            res[i] = a[i] * b[i] - res[i];
            res[i + N / 2] += a[i] * b[i + N / 2];
            res[i + N / 2] += a[i + N / 2] * b[i];
        }
        __syncthreads();
    }

    __global__ void PolyMullvl1(cuPolynomiallvl1 res, const cuPolynomiallvl1 a,
                            const cuPolynomiallvl1 b)
    {
        __shared__ cuPolynomialInFDlvl1 buff[2];
        TwistIFFTlvl1(buff[0], a);
        TwistIFFTlvl1(buff[1], b);
        MulInFD<TFHEpp::DEF_N>(buff[0], buff[0], buff[1]);
        TwistFFTlvl1(res, buff[0]);
    }
}