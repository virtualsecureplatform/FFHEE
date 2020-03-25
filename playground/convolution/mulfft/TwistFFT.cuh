#pragma once
#include"FFTinit.cuh"
#include"coroutines.cuh"

namespace SPCULIOS{
    template<uint32_t N = TFHEpp::DEF_N>
    __device__ inline void TwistMulDirectlvl1(uint32_t* const res, const double* a, const double* twist){
        const unsigned int tid = threadIdx.x;
        const unsigned int bdim = blockDim.x;

        #pragma unroll
        for (int i = tid; i < N / 2; i+=bdim) {
            const double aimbim = a[i + N / 2] * -twist[i + N / 2];
            const double arebim = a[i] * -twist[i + N / 2];
            res[i] = static_cast<int64_t>((a[i] * twist[i] - aimbim)*(2.0/N));
            res[i + N / 2] = static_cast<int64_t>((a[i + N / 2] * twist[i] + arebim)*(2.0/N));
        }
        __threadfence();
    }

    template<uint32_t Nbit = TFHEpp::DEF_Nbit-1>
    __device__ inline void FFT(double* const res, const double* table){
        constexpr uint32_t N = 1<<Nbit;
        constexpr int basebit  = 3;

        const unsigned int tid = threadIdx.x;
        const unsigned int bdim = blockDim.x;

        constexpr uint32_t flag = Nbit%basebit;
        switch(flag){
            case 0:
                for(int index=tid;index<N/8;index+=bdim){
                    double* const res0 = &res[index*8];
                    double* const res1 = res0+1;
                    double* const res2 = res0+2;
                    double* const res3 = res0+3;
                    double* const res4 = res0+4;
                    double* const res5 = res0+5;
                    double* const res6 = res0+6;
                    double* const res7 = res0+7;

                    ButterflyAdd<N>(res0,res1);
                    ButterflyAdd<N>(res2,res3);
                    ButterflyAdd<N>(res4,res5);
                    ButterflyAdd<N>(res6,res7);
                    
                    Radix4TwiddleMul<Nbit,false>(res3);
                    Radix4TwiddleMul<Nbit,false>(res7);
                    
                    ButterflyAdd<N>(res0,res2);
                    ButterflyAdd<N>(res1,res3);
                    ButterflyAdd<N>(res4,res6);
                    ButterflyAdd<N>(res5,res7);

                    Radix8TwiddleMulStrideOne<Nbit,false>(res5);
                    Radix4TwiddleMul<Nbit,false>(res6);
                    Radix8TwiddleMulStrideThree<Nbit,false>(res7);

                    ButterflyAdd<N>(res0,res4);
                    ButterflyAdd<N>(res1,res5);
                    ButterflyAdd<N>(res2,res6);
                    ButterflyAdd<N>(res3,res7);
                }
                break;
            case 2:
                for(int index=tid;index<N/4;index+=bdim){
                    double* const res0 = &res[index*4];
                    double* const res1 = res0+1;
                    double* const res2 = res0+2;
                    double* const res3 = res0+3;

                    ButterflyAdd<N>(res0,res1);
                    ButterflyAdd<N>(res2,res3);

                    Radix4TwiddleMul<Nbit,false>(res3);

                    ButterflyAdd<N>(res0,res2);
                    ButterflyAdd<N>(res1,res3);

                }
                break;
            case 1:
                for(int index=tid;index<N/2;index+=bdim){
                    double* const res0 = &res[index*2];
                    double* const res1 = res0+1;

                    ButterflyAdd<N>(res0,res1);
                }
                break;
        }
        __syncthreads();

        #pragma unroll
        for(int step = Nbit-(flag>0?flag:basebit)-basebit; step>=0; step-=basebit){
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
                
                TwiddleMul<Nbit,4,false>(res1,table,elementindex,step);
                TwiddleMul<Nbit,2,false>(res2,table,elementindex,step);
                TwiddleMul<Nbit,6,false>(res3,table,elementindex,step);
                TwiddleMul<Nbit,1,false>(res4,table,elementindex,step);
                TwiddleMul<Nbit,5,false>(res5,table,elementindex,step);
                TwiddleMul<Nbit,3,false>(res6,table,elementindex,step);
                TwiddleMul<Nbit,7,false>(res7,table,elementindex,step);
                            
                ButterflyAdd<N>(res0,res1);
                ButterflyAdd<N>(res2,res3);
                ButterflyAdd<N>(res4,res5);
                ButterflyAdd<N>(res6,res7);
                
                Radix4TwiddleMul<Nbit,false>(res3);
                Radix4TwiddleMul<Nbit,false>(res7);

                ButterflyAdd<N>(res0,res2);
                ButterflyAdd<N>(res1,res3);
                ButterflyAdd<N>(res4,res6);
                ButterflyAdd<N>(res5,res7);
                
                Radix8TwiddleMulStrideOne<Nbit,false>(res5);
                Radix4TwiddleMul<Nbit,false>(res6);
                Radix8TwiddleMulStrideThree<Nbit,false>(res7);

                ButterflyAdd<N>(res0,res4);
                ButterflyAdd<N>(res1,res5);
                ButterflyAdd<N>(res2,res6);
                ButterflyAdd<N>(res3,res7);
            }
            __syncthreads();
        }
    }

    __device__ inline void TwistFFTlvl1(uint32_t* const res, double* a){
        FFT<TFHEpp::DEF_Nbit-1>(a,tablelvl1);
        TwistMulDirectlvl1<TFHEpp::DEF_N>(res,a,twistlvl1);
    }
}