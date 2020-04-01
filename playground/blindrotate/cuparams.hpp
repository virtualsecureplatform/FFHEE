#pragma once
#include <array>
#include <cmath>
#include <cstdint>

namespace FFHEE{
    using cuKeylvl1 = uint32_t[TFHEpp::DEF_N];
    using cuKeylvl2 = uint64_t[TFHEpp::DEF_nbar];

    using cuTLWElvl0 = uint32_t[TFHEpp::DEF_n + 1];
    using cuTLWElvl1 = uint32_t[TFHEpp::DEF_N + 1];
    using cuTLWElvl2 = uint64_t[TFHEpp::DEF_nbar + 1];

    using cuPolynomiallvl1 = uint32_t[TFHEpp::DEF_N];
    using cuPolynomiallvl2 = uint64_t[TFHEpp::DEF_nbar];
    using cuPolynomialInFDlvl1 = double[TFHEpp::DEF_N];
    using cuPolynomialInFDlvl2 = double[TFHEpp::DEF_nbar];

    using cuTRLWElvl1 = cuPolynomiallvl1[2];
    using cuTRLWElvl2 = cuPolynomiallvl2[2];
    using cuTRLWEInFDlvl1 = cuPolynomialInFDlvl1[2];
    using cuTRLWEInFDlvl2 = cuPolynomialInFDlvl2[2];
    using cuDecomposedTRLWElvl1 = cuPolynomiallvl1[2 * TFHEpp::DEF_l];
    using cuDecomposedTRLWElvl2 = cuPolynomiallvl2[2 * TFHEpp::DEF_lbar];
    using cuDecomposedTRLWEInFDlvl1 = cuPolynomialInFDlvl1[2 * TFHEpp::DEF_l];
    using cuDecomposedTRLWEInFDlvl2 = cuPolynomialInFDlvl2[2 * TFHEpp::DEF_lbar];

    using cuTRGSWlvl1 = cuTRLWElvl1[2 * TFHEpp::DEF_l];
    using cuTRGSWlvl2 = cuTRLWElvl2[2 * TFHEpp::DEF_lbar];
    using cuTRGSWFFTlvl1 = double[2 * TFHEpp::DEF_l][2][TFHEpp::DEF_N];
    using cuTRGSWFFTlvl2 = cuTRLWEInFDlvl2[2 * TFHEpp::DEF_lbar];

    using cuBootStrappingKeyFFTlvl01 = cuTRGSWFFTlvl1[TFHEpp::DEF_n];
    using cuBootStrappingKeyFFTlvl02 = cuTRGSWFFTlvl2[TFHEpp::DEF_n];

    using cuKeySwitchingKey =
        cuTLWElvl0[TFHEpp::DEF_N][TFHEpp::DEF_t][(1 << TFHEpp::DEF_basebit) - 1];
    using cuPrivKeySwitchKey =
        cuTRLWElvl1[TFHEpp::DEF_nbar][TFHEpp::DEF_tbar][(1 << TFHEpp::DEF_basebitlvl21) - 1];
}