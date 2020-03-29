#include <random>
#include <cassert>
#include <iostream>
#include <cmath>
#include "cuparams.hpp"
#include "params.hpp"

using namespace TFHEpp;
using namespace FFHEE;
using namespace std;

__device__ uint32_t d_res[DEF_N];
__device__ uint32_t d_a[DEF_N];

__global__ void Copy(){
    for(int i = threadIdx.x;i<DEF_N;i+=blockDim.x)d_res[i]=d_a[i];
    __threadfence();
}


int main( int argc, char** argv) 
{
    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint32_t> Torus32dist(0, UINT32_MAX);

    Polynomiallvl1 a;
    for (uint32_t &i : a) i = Torus32dist(engine);

    Polynomiallvl1 res;
    cudaHostRegister(a.data(),sizeof(a),cudaHostRegisterDefault);
    cudaHostRegister(res.data(),sizeof(res),cudaHostRegisterDefault);


    cudaMemcpyToSymbolAsync(d_a,a.data(),sizeof(a),0,cudaMemcpyHostToDevice );
    Copy<<<1,DEF_N>>>();
    cudaMemcpyFromSymbolAsync(res.data(),d_res,sizeof(res),0,cudaMemcpyDeviceToHost );

    cudaDeviceSynchronize();

    for(int i = 0;i<DEF_N;i++) cout<<i<<":"<<res[i]<<":"<<a[i]<<endl;
    for(int i = 0;i<DEF_N;i++) assert(res[i]==a[i]);
    cout<<"PASS"<<endl;
}