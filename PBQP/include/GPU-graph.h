#pragma once 

#include "PBQP.h"

namespace device {

struct Graph final {
  using Cost_t = float;
  using Index_t = int;

private:
  // AdjMatrix[i][j] - index in the cost records or -1 
  //  if there is no edge between nodes
  // AdjMatrix[i][i] - cost vector of a node i
  device::Matrix<Index_t> AdjMatrix;
  device::Matrix<Cost_t> *Costs = nullptr;
  unsigned NumOfCosts = 0;

  // This vector stores cuda memory to be free
  std::vector<device::Matrix<Cost_t>> CostMatrices;

public:
  __host__
  Graph() = default;

  __host__
  Graph(const PBQP::Graph &HostGraph);

  __host__
  void free();
};

} // namespace device