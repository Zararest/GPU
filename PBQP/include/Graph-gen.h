#pragma once

#include "PBQP.h"

namespace PBQP {

struct GenConfig {
  size_t NumOfNodes = 8;
  size_t NumOfCliques = 1;
  size_t VectSize = 8;
  double AvgNeighbNum = 3;
  bool HasInfCosts = false;
};

Graph generateGraph(GenConfig Cfg);

} // namespace PBQP