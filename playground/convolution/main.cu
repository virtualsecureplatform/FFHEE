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
    uniform_int_distribution<uint32_t> Bgdist(0, DEF_Bg);

    Polynomiallvl1 a,b,h_res,res;
    for (uint32_t &i : a) i = Torus32dist(engine);
    for (uint32_t &i : b) i = Bgdist(engine);

    SPQLIOSpp::PolyMullvl1(h_res, a, b);

    uint32_t* d_a,*d_b,*d_res;
    cudaMalloc( (void**) &d_a, sizeof(a));
    cudaMalloc( (void**) &d_b, sizeof(b));
    cudaMalloc( (void**) &d_res, sizeof(res));
    cudaMemcpy(d_a,a.data(),sizeof(a),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b.data(),sizeof(b),cudaMemcpyHostToDevice);
    FFTinit();
    PolyMullvl1<<<1,64>>>(d_res,d_a,d_b);
    cudaMemcpy(res.data(),d_res,sizeof(res),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i = 0;i<DEF_N;i++) {cout<<i<<":"<<res[i]<<":"<<h_res[i]<<endl;}
    for(int i = 0;i<DEF_N;i++) {assert(abs(static_cast<int32_t>(res[i]-h_res[i]))<=1);}
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
    cout<<"PASS"<<endl;
}