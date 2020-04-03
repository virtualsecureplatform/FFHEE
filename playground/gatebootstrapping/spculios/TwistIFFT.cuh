#pragma once
#include"FFTinit.cuh"
#include"coroutines.cuh"

namespace SPCULIOS{
    using namespace FFHEE;
    
    template<typename T = uint32_t,uint32_t N = TFHEpp::DEF_N>
    __device__ inline void TwistMulInvert(double* const res, const T* a, const double* twist){
        const unsigned int tid = threadIdx.x;
        const unsigned int bdim = blockDim.x;

        #pragma unroll
        for (int i = tid; i < N / 2; i+=bdim) {
            const double are = static_cast<double>(static_cast<typename std::make_signed<T>::type>(a[i]));
            const double aim = static_cast<double>(static_cast<typename std::make_signed<T>::type>(a[i+N/2]));
            const double aimbim = aim * twist[i + N / 2];
            const double arebim = are * twist[i + N / 2];
            res[i] = are * twist[i] - aimbim;
            res[i + N / 2] = aim * twist[i] + arebim;
        }
        __syncthreads();
    }

    template<uint32_t N = TFHEpp::DEF_N>
    __device__ inline void TwistMulInvertinPlace(double* const a, const double* twist){
        const unsigned int tid = threadIdx.x;
        const unsigned int bdim = blockDim.x;

        #pragma unroll
        for (int i = tid; i < N / 2; i+=bdim) {
            const double are = a[i];
            const double aim = a[i+N/2];
            const double aimbim = aim * twist[i + N / 2];
            const double arebim = are * twist[i + N / 2];
            a[i] = are * twist[i] - aimbim;
            a[i + N / 2] = aim * twist[i] + arebim;
        }
        __syncthreads();
    }
    
    template<uint32_t Nbit = TFHEpp::DEF_Nbit-1>
    __device__ inline void IFFT(double* const res, const double* table){
        constexpr uint32_t N = 1<<Nbit;
        constexpr uint32_t basebit  = 3;

        const unsigned int tid = threadIdx.x;
        const unsigned int bdim = blockDim.x;

        #pragma unroll
        for(int step = 0; step+basebit<Nbit-1; step+=basebit){
            const uint32_t size = 1<<(Nbit-step);
            const uint32_t elementmask = (size>>basebit)-1;

            #pragma unroll
            for(int index=tid;index<(N>>basebit);index+=bdim){
                const uint32_t elementindex = index&elementmask;
                const uint32_t blockindex = (index - elementindex)>>(Nbit-step-basebit);

                double* const res0 = &res[blockindex*size+elementindex];
                double* const res1 = res0+size/8;
                double* const res2 = res0+2*size/8;
                double* const res3 = res0+3*size/8;
                double* const res4 = res0+4*size/8;
                double* const res5 = res0+5*size/8;
                double* const res6 = res0+6*size/8;
                double* const res7 = res0+7*size/8;

                ButterflyAdd<N>(res0,res4);
                ButterflyAdd<N>(res1,res5);
                ButterflyAdd<N>(res2,res6);
                ButterflyAdd<N>(res3,res7);

                Radix8TwiddleMulStrideOne<Nbit,true>(res5);
                Radix4TwiddleMul<Nbit,true>(res6);
                Radix8TwiddleMulStrideThree<Nbit,true>(res7);

                ButterflyAdd<N>(res0,res2);
                ButterflyAdd<N>(res1,res3);
                ButterflyAdd<N>(res4,res6);
                ButterflyAdd<N>(res5,res7);

                Radix4TwiddleMul<Nbit,true>(res3);
                Radix4TwiddleMul<Nbit,true>(res7);
                
                ButterflyAdd<N>(res0,res1);
                ButterflyAdd<N>(res2,res3);
                ButterflyAdd<N>(res4,res5);
                ButterflyAdd<N>(res6,res7);

                TwiddleMul<Nbit,4,true>(res1,table,elementindex,step);
                TwiddleMul<Nbit,2,true>(res2,table,elementindex,step);
                TwiddleMul<Nbit,6,true>(res3,table,elementindex,step);
                TwiddleMul<Nbit,1,true>(res4,table,elementindex,step);
                TwiddleMul<Nbit,5,true>(res5,table,elementindex,step);
                TwiddleMul<Nbit,3,true>(res6,table,elementindex,step);
                TwiddleMul<Nbit,7,true>(res7,table,elementindex,step);
            }
            __syncthreads();
        }

        constexpr uint32_t flag = Nbit%basebit;
        switch(flag){
            case 0:
                #pragma unroll
                for(int index=tid;index<N/8;index+=bdim){
                    double* const res0 = &res[index*8];
                    double* const res1 = res0+1;
                    double* const res2 = res0+2;
                    double* const res3 = res0+3;
                    double* const res4 = res0+4;
                    double* const res5 = res0+5;
                    double* const res6 = res0+6;
                    double* const res7 = res0+7;

                    ButterflyAdd<N>(res0,res4);
                    ButterflyAdd<N>(res1,res5);
                    ButterflyAdd<N>(res2,res6);
                    ButterflyAdd<N>(res3,res7);

                    Radix8TwiddleMulStrideOne<Nbit,true>(res5);
                    Radix4TwiddleMul<Nbit,true>(res6);
                    Radix8TwiddleMulStrideThree<Nbit,true>(res7);

                    ButterflyAdd<N>(res0,res2);
                    ButterflyAdd<N>(res1,res3);
                    ButterflyAdd<N>(res4,res6);
                    ButterflyAdd<N>(res5,res7);

                    Radix4TwiddleMul<Nbit,true>(res3);
                    Radix4TwiddleMul<Nbit,true>(res7);

                    ButterflyAdd<N>(res0,res1);
                    ButterflyAdd<N>(res2,res3);
                    ButterflyAdd<N>(res4,res5);
                    ButterflyAdd<N>(res6,res7);
                }
                break;
            case 2:
                #pragma unroll
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
                break;
            case 1:
                #pragma unroll
                for(int index=tid;index<N/2;index+=bdim){
                    double* const res0 = &res[index*2];
                    double* const res1 = res0+1;

                    ButterflyAdd<N>(res0,res1);
                }
                break;
        }
        __syncthreads();
    }

    __device__ inline void TwistIFFTlvl1(cuPolynomialInFDlvl1 res, const cuPolynomiallvl1 a){
            TwistMulInvert<uint32_t,TFHEpp::DEF_N>(res,a,twistlvl1);
            IFFT<TFHEpp::DEF_Nbit-1>(res,tablelvl1);
        }
    
        __device__ inline void TwistIFFTinPlacelvl1(cuPolynomialInFDlvl1 a,const double* twist,const double* table){
            TwistMulInvertinPlace<TFHEpp::DEF_N>(a,twist);
            IFFT<TFHEpp::DEF_Nbit-1>(a,table);
        }

}