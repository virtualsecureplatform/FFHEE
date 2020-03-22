#include"spqlios++.hpp"
#include <random>
#include <cassert>
#include <iostream>
#include <cmath>

using namespace TFHEpp;
using namespace std;

template<typename T = uint32_t,uint32_t N = DEF_N>
__global__ void cudaTwistMulInvert(double* res, T* a, const double* twist){
    unsigned int tid = threadIdx.x;
    unsigned int bdim = blockDim.x;
        for (int i = tid; i < N / 2; i+=bdim) {
            const double are = static_cast<double>(static_cast<typename make_signed<T>::type>(a[i]));
            const double aim = static_cast<double>(static_cast<typename make_signed<T>::type>(a[i+N/2]));
            const double aimbim = aim * twist[i + N / 2];
            const double arebim = are * twist[i + N / 2];
            res[i] = are * twist[i] - aimbim;
            res[i + N / 2] = aim * twist[i] + arebim;
        }
        __threadfence();
    }

int main( int argc, char** argv) 
{
    const array<double,DEF_N> h_twistlvl1 = SPQLIOSpp::TwistGen<DEF_N>();
    double* twistlvl1;
    cudaMalloc( (void**) &twistlvl1, sizeof(h_twistlvl1));
    cudaMemcpy( twistlvl1, h_twistlvl1.data(), sizeof(h_twistlvl1),cudaMemcpyHostToDevice);

    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint32_t> Torus32dist(0, UINT32_MAX);

    Polynomiallvl1 a;
    for (uint32_t &i : a) i = Torus32dist(engine);

    PolynomialInFDlvl1 h_res,res;
    SPQLIOSpp::TwistMulInvert<uint32_t,DEF_N>(h_res,a,h_twistlvl1);

    uint32_t* d_a;
    double* d_res;
    cudaMalloc( (void**) &d_a, sizeof(a));
    cudaMalloc( (void**) &d_res, sizeof(res));
    cudaMemcpy(d_a,a.data(),sizeof(a),cudaMemcpyHostToDevice);
    cudaTwistMulInvert<<<1,16>>>(d_res,d_a,twistlvl1);
    cudaMemcpy(res.data(),d_res,sizeof(res),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i = 0;i<DEF_N;i++) assert(abs(res[i]-h_res[i])<1e-6);
    cudaFree(d_a);
    cudaFree(d_res);
    cudaFree(twistlvl1);
    cout<<"PASS"<<endl;
}