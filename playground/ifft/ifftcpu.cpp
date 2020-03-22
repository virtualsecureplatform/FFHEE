#include"spqlios++.hpp"
#include <random>
#include <cassert>
#include <iostream>
#include <cmath>
#include <math.h>

using namespace TFHEpp;
using namespace std;


template<typename T = uint32_t,uint32_t N = DEF_N>
void TwistMulInvert(double* res, const T* a, const double* twist){
        for (int i = 0; i < N / 2; i+=1) {
            const double are = static_cast<double>(static_cast<typename make_signed<T>::type>(a[i]));
            const double aim = static_cast<double>(static_cast<typename make_signed<T>::type>(a[i+N/2]));
            const double aimbim = aim * twist[i + N / 2];
            const double arebim = are * twist[i + N / 2];
            res[i] = are * twist[i] - aimbim;
            res[i + N / 2] = aim * twist[i] + arebim;
        }
    }

template<uint32_t N>
void ButterflyAdd(double* const a, double* const b){
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

template <uint32_t Nbit = DEF_Nbit-1, uint32_t stride, bool isinvert =true>
void TwiddleMul(double* const a, const double* table,const int i,const int step){
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

template <uint32_t Nbit = DEF_Nbit-1, bool isinvert =true>
void Radix4TwiddleMul(double* const a){
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

template <uint32_t Nbit = DEF_Nbit-1, bool isinvert =true>
void Radix8TwiddleMulStrideOne(double* const a){
    constexpr uint32_t N = 1<<Nbit;

    double& are = *a;
    double& aim = *(a+N);
    const double aimbim = isinvert?aim:-aim;
    const double arebim = isinvert?are:-are;
    are = M_SQRT1_2 *(are - aimbim);
    aim = M_SQRT1_2 *(aim + arebim);
}

template <uint32_t Nbit = DEF_Nbit-1, bool isinvert =true>
void Radix8TwiddleMulStrideThree(double* const a){
    constexpr uint32_t N = 1<<Nbit;

    double& are = *a;
    double& aim = *(a+N);
    const double aimbim = isinvert?aim:-aim;
    const double arebim = isinvert?are:-are;
    are = M_SQRT1_2 *(-are - aimbim);
    aim = M_SQRT1_2 *(-aim + arebim);
}

template<uint32_t Nbit = DEF_Nbit-1>
void IFFT(double* const res, const double* table){
    constexpr uint32_t N = 1<<Nbit;
    constexpr uint32_t basebit  = 3;

    for(int step = 0; step+basebit<Nbit-1; step+=basebit){
        const uint32_t size = 1<<(Nbit-step);
        const uint32_t elementmask = (size>>basebit)-1;
        for(int index=0;index<(N>>basebit);index+=1){
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
    }

    constexpr uint32_t flag = Nbit%basebit;
    switch(flag){
        case 0:
            for(int index=0;index<N/8;index+=1){
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
            for(int index=0;index<N/4;index+=1){
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
            for(int index=0;index<N/2;index+=1){
                double* const res0 = &res[index*2];
                double* const res1 = res0+1;

                ButterflyAdd<N>(res0,res1);
            }
            break;
    }
}

void TwistIFFTlvl1(double* const res, const uint32_t* a, const double* twistlvl1, const double* tablelvl1){
        TwistMulInvert<uint32_t,DEF_N>(res,a,twistlvl1);
        IFFT<DEF_Nbit-1>(res,tablelvl1);
    }

int main( int argc, char** argv) 
{
    const array<double,DEF_N> h_twistlvl1 = SPQLIOSpp::TwistGen<DEF_N>();
    const array<double,DEF_N> h_tablelvl1 = SPQLIOSpp::TableGen<DEF_N/2>();
    const double* twistlvl1,*tablelvl1;
    twistlvl1 = h_twistlvl1.data();
    tablelvl1 = h_tablelvl1.data();

    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint32_t> Torus32dist(0, UINT32_MAX);

    Polynomiallvl1 a;
    for (uint32_t &i : a) i = Torus32dist(engine);

    PolynomialInFDlvl1 h_res;
    SPQLIOSpp::TwistIFFTlvl1(h_res,a);

    uint32_t* d_a;
    double res[DEF_N];
    d_a = a.data();
    TwistIFFTlvl1(res,d_a,twistlvl1,tablelvl1);
    //for(int i = 0;i<DEF_N;i++) {cout<<i<<":"<<res[i]<<":"<<h_res[i]<<endl;}
    for(int i = 0;i<DEF_N;i++) {assert(abs(res[i]-h_res[i])<1);}
    cout<<"PASS"<<endl;
}