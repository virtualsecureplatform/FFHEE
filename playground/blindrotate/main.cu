#include<tfhe++.hpp>
#include<params.hpp>
#include <random>
#include <cassert>
#include <iostream>
#include <cmath>
#include "gatebootstrapping.cuh"

using namespace FFHEE;
using namespace SPCULIOS;
using namespace std;

template <typename T = uint32_t, uint32_t N = TFHEpp::DEF_N>
inline void PolynomialMulByXai(array<T, N> &res, const array<T, N> &poly,
                               const T a)
{
    if (a == 0)
        return;
    else if (a < N) {
        for (int i = 0; i < a; i++) res[i] = -poly[i - a + N];
        for (int i = a; i < N; i++) res[i] = poly[i - a];
    }
    else {
        const T aa = a - N;
        for (int i = 0; i < aa; i++) res[i] = poly[i - aa + N];
        for (int i = aa; i < N; i++) res[i] = -poly[i - aa];
    }
}

int main( int argc, char** argv) 
{

    using namespace TFHEpp;
    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint32_t> binary(0, 1);
    uniform_int_distribution<uint32_t> rotate(0, 2*DEF_N-1);

    lweKey key;

    array<bool, DEF_N> p;
    for (bool &i : p) i = (binary(engine) > 0);
    Polynomiallvl1 pmu;
    for (int i = 0; i < DEF_N; i++) pmu[i] = p[i] ? DEF_μ : -DEF_μ;
    TRLWElvl1 c = trlweSymEncryptlvl1(pmu, DEF_αbk, key.lvl1);
    TRLWElvl1 res,h_res;

    uint32_t s = binary(engine);
    uint32_t a = rotate(engine);

    if(s>0){
        PolynomialMulByXai<uint32_t,DEF_N>(h_res[0],c[0],a);
        PolynomialMulByXai<uint32_t,DEF_N>(h_res[1],c[1],a);
    }else{
        h_res = c;
    }

    TRGSWFFTlvl1 trgswfft = trgswfftSymEncryptlvl1(s, DEF_αbk, key.lvl1);

    FFTinit();
    FFHEE::BlindRotateFFTlvl1(res,c,trgswfft,a);
    cudaDeviceSynchronize();

    array<bool, DEF_N> p2 = trlweSymDecryptlvl1(res, key.lvl1);
    array<bool, DEF_N> resp = trlweSymDecryptlvl1(h_res, key.lvl1);
    
    for(int i = 0;i<DEF_N;i++) {cout<<i<<":"<<p2[i]<<":"<<resp[i]<<endl;}
    cout<<s<<endl;
    for (int i = 0; i < DEF_N; i++) assert(resp[i] == p2[i]);
    cout<<"PASS"<<endl;
}