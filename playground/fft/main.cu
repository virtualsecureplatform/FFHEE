#include"spqlios++.hpp"
#include <random>
#include <cassert>
#include <iostream>
#include <cmath>
#include <mulfft/mulfft.cuh>

using namespace TFHEpp;
using namespace SPCULIOS;
using namespace std;

int main( int argc, char** argv) 
{

    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint32_t> Torus32dist(0, UINT32_MAX);

    Polynomiallvl1 a,res;
    for (uint32_t &i : a) i = Torus32dist(engine);


    uint32_t* d_a;
    double* d_res;
    cudaMalloc( (void**) &d_a, sizeof(a));
    cudaMalloc( (void**) &d_res, sizeof(PolynomialInFDlvl1));
    cudaMemcpy(d_a,a.data(),sizeof(a),cudaMemcpyHostToDevice);
    FFTinit();
    TwistIFFTlvl1<<<1,64>>>(d_res,d_a);
    TwistFFTlvl1<<<1,64>>>(d_a,d_res);
    cudaMemcpy(res.data(),d_a,sizeof(res),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i = 0;i<DEF_N;i++) {cout<<i<<":"<<res[i]<<":"<<a[i]<<endl;}
    for(int i = 0;i<DEF_N;i++) {assert(abs(static_cast<int32_t>(res[i]-a[i]))<=1);}
    cudaFree(d_a);
    cudaFree(d_res);
    cudaFree(twistlvl1);
    cudaFree(tablelvl1);
    cout<<"PASS"<<endl;
}