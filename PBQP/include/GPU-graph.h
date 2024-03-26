#pragma once 

#include "PBQP.h"

namespace device {

struct Graph final {
  using Cost_t = float;
  using Index_t = int;

private:
  static constexpr Index_t NoEdge = -1;
  // AdjMatrix[i][j] - index in the cost records or -1 
  //  if there is no edge between nodes
  // AdjMatrix[i][i] - cost vector of a node i
  device::Matrix<Index_t> AdjMatrix;
  host::Matrix<Index_t> HostAdjMatrix;
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

  __host__
  size_t getNumOfCostCombinations() const;

  __host__
  size_t getNodeCostSize(unsigned NodeIdx) const {
    assert(NodeIdx < HostAdjMatrix.h());
    auto CostIdx = HostAdjMatrix[NodeIdx][NodeIdx];
    return CostMatrices[CostIdx].h();
  }

  __host__
  size_t size() const {
    assert(HostAdjMatrix.h() == HostAdjMatrix.w());
    return HostAdjMatrix.h();
  }

  __device__
  device::Matrix<Index_t> &getAdjMatrix() {
    return AdjMatrix;
  }

  __device__ 
  device::Matrix<Cost_t> &getCostMatrix(Index_t Index) {
    assert(Index >= 0);
    assert(Index < NumOfCosts);
    return Costs[Index];
  }

  __device__ 
  unsigned getNumOfCosts() const {
    return NumOfCosts;
  }
};

} // namespace device