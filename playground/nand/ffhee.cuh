#pragma once
#include <params.hpp>
#include <cloudkey.hpp>

namespace FFHEE{
void cuHomNAND(TFHEpp::TLWElvl0 &res, const TFHEpp::TLWElvl0 &ca, const TFHEpp::TLWElvl0 &cb,
             const TFHEpp::GateKey &gk);
}