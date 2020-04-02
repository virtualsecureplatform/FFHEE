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
    constexpr uint32_t num_test = 1000;

    using namespace TFHEpp;
    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint32_t> binary(0, 1);

    SecretKey sk;
    GateKey* gk = new GateKey(sk);

    FFTinit();

    for (int test = 0; test < num_test; test++) {
        bool p = binary(engine) > 0;
        TLWElvl0 tlwe =
            tlweSymEncryptlvl0(p ? DEF_μ : -DEF_μ, DEF_α, sk.key.lvl0);
        TLWElvl0 bootedtlwe;

        FFHEE::GateBootstrapping(bootedtlwe, tlwe, *gk);
        cudaDeviceSynchronize();

        bool p2 = tlweSymDecryptlvl0(bootedtlwe, sk.key.lvl0);
        assert(p == p2);
    }
    cout<<"PASS"<<endl;
}