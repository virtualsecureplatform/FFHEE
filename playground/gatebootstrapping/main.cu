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
    constexpr uint32_t num_test = NUMBER_OF_STREAM;
    array<cudaStream_t,NUMBER_OF_STREAM> starray;
    for(cudaStream_t &st:starray) cudaStreamCreate(&st);

    using namespace TFHEpp;
    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint32_t> binary(0, 1);

    SecretKey sk;
    GateKey* gk = new GateKey(sk);

    FFHEEinit(*gk);

    array<uint8_t,num_test> parray;
    for(uint8_t &p:parray) p = binary(engine);
    array<uint32_t,num_test> muarray;
    for(int i = 0;i<num_test;i++) muarray[i] = (parray[i]>0)? DEF_μ : -DEF_μ;
    array<TLWElvl0,num_test> tlwearray,bootedtlwearray;
    for(int i = 0;i<num_test;i++) {
        tlwearray[i]=tlweSymEncryptlvl0(muarray[i], DEF_α, sk.key.lvl0);
        cudaHostRegister(tlwearray[i].data(), sizeof(tlwearray[i]),cudaHostRegisterDefault);
        cudaHostRegister(bootedtlwearray[i].data(), sizeof(bootedtlwearray[i]),cudaHostRegisterDefault);
    }

    float et;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int test = 0; test < num_test; test++) {
        FFHEE::GateBootstrapping(bootedtlwearray[test], tlwearray[test], starray[test],test);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    cout<<"Total Time:"<<et<<"ms"<<endl;
    cout<<"Per Gate:"<<et/num_test<<"ms"<<endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for(int i = 0; i < num_test; i++ )assert(parray[i] == tlweSymDecryptlvl0(bootedtlwearray[i], sk.key.lvl0));
    cout<<"PASS"<<endl;
    for(cudaStream_t &st:starray) cudaStreamDestroy(st);
}