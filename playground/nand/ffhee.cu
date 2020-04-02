#include"ffhee.cuh"
#include"gatebootstrapping.cuuh"
#include"spculios/spculios.cuh"

namespace FFHEE{

__device__ cuBootStrappingKeyFFTlvl01 d_bkfftlvl01;
__device__ cuKeySwitchingKey d_ksk;
__device__ cuTLWElvl0 d_ca, d_cb, d_res;

void FFHEEinit(TFHEpp::GateKey &gk){
    FFTinit();
    cudaMemcpyToSymbol(d_bkfftlvl01,gk.bkfftlvl01.data(),sizeof(gk.bkfftlvl01));
    cudaMemcpyToSymbol(d_ksk,gk.ksk.data(),sizeof(gk.ksk));
}

void cuHomNAND(TFHEpp::TLWElvl0 &res, const TFHEpp::TLWElvl0 &ca, const TFHEpp::TLWElvl0 &cb,
             const TFHEpp::GateKey &gk){
    cudaMemcpyToSymbolAsync(d_ca,ca.data(),sizeof(ca));
    cudaMemcpyToSymbolAsync(d_cb,ca.data(),sizeof(cb));
}
}