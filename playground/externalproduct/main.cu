#include<tfhe++.hpp>
#include<params.hpp>
#include <random>
#include <cassert>
#include <iostream>
#include <cmath>
#include "externalproduct.cuh"

using namespace FFHEE;
using namespace SPCULIOS;
using namespace std;

int main( int argc, char** argv) 
{

    using namespace TFHEpp;
    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint32_t> binary(0, 1);

    lweKey key;

    array<bool, DEF_N> p;
    for (bool &i : p) i = (binary(engine) > 0);
    Polynomiallvl1 pmu;
    for (int i = 0; i < DEF_N; i++) pmu[i] = p[i] ? DEF_μ : -DEF_μ;
    TRLWElvl1 c = trlweSymEncryptlvl1(pmu, DEF_αbk, key.lvl1);
    TRLWElvl1 res;

    TRGSWFFTlvl1 trgswfft = trgswfftSymEncryptlvl1(1, DEF_αbk, key.lvl1);

    FFTinit();
    FFHEE::trgswfftExternalProductlvl1(res,c,trgswfft);
    cudaDeviceSynchronize();

    array<bool, DEF_N> p2 = trlweSymDecryptlvl1(res, key.lvl1);
    
    for(int i = 0;i<DEF_N;i++) {cout<<i<<":"<<p2[i]<<":"<<p[i]<<endl;}
    for (int i = 0; i < DEF_N; i++) assert(p[i] == p2[i]);
    cout<<"PASS"<<endl;
}