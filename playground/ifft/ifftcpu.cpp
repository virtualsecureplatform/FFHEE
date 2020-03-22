#include"spqlios++.hpp"
#include <random>
#include <cassert>
#include <iostream>
#include <cmath>

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


template<uint32_t Nbit = DEF_Nbit-1>
void IFFT(double* const res, const double* table){
    constexpr uint32_t N = 1<<Nbit;
    constexpr uint32_t basebit  = 1;

    for(int step = 0; step<Nbit-basebit; step+=basebit){
        const uint32_t size = 1<<(Nbit-step);
        const uint32_t elementmask = (size>>basebit)-1;
        for(int index=0;index<(N>>basebit);index+=1){
            const uint32_t elementindex = index&elementmask;
            const uint32_t blockindex = (index - elementindex)>>(Nbit-step-basebit);
            double* const res0 = &res[blockindex*size+elementindex];
            double* const res1 = res0+size/2;
            cout<<index<<":"<<elementindex<<":"<<blockindex<<":"<<size<<":"<<blockindex*size+elementindex<<endl;
            ButterflyAdd<N>(res0,res1);
            TwiddleMul<Nbit,1,true>(res1,table,elementindex,step);
        }
    }

    constexpr uint32_t size = 2;
    constexpr uint32_t elementindex = 0;
        
    for(int index=0;index<N/2;index+=1){
        const uint32_t blockindex = index - elementindex;
        double* const res0 = &res[blockindex*size+elementindex];
        double* const res1 = res0+size/2;

        ButterflyAdd<N>(res0,res1);
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
    for(int i = 0;i<DEF_N;i++) {cout<<i<<":"<<res[i]<<":"<<h_res[i]<<endl;assert(abs(res[i]-h_res[i])<1);}
    cout<<"PASS"<<endl;
}