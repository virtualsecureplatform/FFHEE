#include<params.hpp>
#include<mulfft.cuh>
#include <random>

using namespace TFHEpp;

void main(){
    const uint32_t num_test = 1000;
    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint32_t> Torus32dist(0, UINT32_MAX);

    cout << "Start LVL1 test." << endl;
    for (int test; test < num_test; test++) {
        Polynomiallvl1 a,res;
        for (uint32_t &i : a) i = Torus32dist(engine);
        FFHEE::FFTlvl1Test(res,a);
        for (int i = 0; i < DEF_N; i++)
            assert(abs(static_cast<int32_t>(a[i] - res[i])) <= 1);
    }
    cout << "FFT Passed" << endl
}