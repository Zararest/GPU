#include "GPU-graph.h"

#include <map>

namespace {
template <typename T>
T *copyVectorToCuda(const std::vector<T> &Vect) {
  T *CostsPtr = nullptr;
  CUDA_CHECK(cudaMalloc((void **)&CostsPtr, Vect.size() * sizeof(T)));
  for (size_t Idx = 0; Idx < Vect.size(); ++Idx)
    CUDA_CHECK(cudaMemcpy(&CostsPtr[Idx], &Vect[Idx], sizeof(T),
                          cudaMemcpyHostToDevice));
  return CostsPtr;
}

} // anonymous namespace

namespace device {

__host__
Graph::Graph(const PBQP::Graph &HostGraph) {
  auto NodeAddrToIdx = std::map<PBQP::Graph::Node*, size_t>{};
  auto Idx = 0ull;
  for (auto &NodePtr : utils::makeRange(HostGraph.nodesBeg(), 
                                        HostGraph.nodesEnd())) {
    NodeAddrToIdx[NodePtr.get()] = Idx;
    ++Idx;
  }

  auto GetIndexes = [&NodeAddrToIdx](auto *Lhs, auto *Rhs) {
    assert(NodeAddrToIdx.find(Lhs) != NodeAddrToIdx.end());
    assert(NodeAddrToIdx.find(Rhs) != NodeAddrToIdx.end());
    return std::make_pair(NodeAddrToIdx[Lhs], NodeAddrToIdx[Rhs]);
  };

  HostAdjMatrix = host::Matrix<Index_t>{HostGraph.size(), 
                                        HostGraph.size()};
  std::fill(HostAdjMatrix.begin(), HostAdjMatrix.end(), NoEdge);
  for (auto &Edge : utils::makeRange(HostGraph.edgesBeg(), 
                                     HostGraph.edgesEnd())) {
    auto [Lhs, Rhs] = Edge->getNodes();
    auto [LhsIdx, RhsIdx] = GetIndexes(Lhs, Rhs);
    fillAdjMatrix(LhsIdx, RhsIdx, CostMatrices.size());
    CostMatrices.emplace_back(Edge->getCostMatrix());
  }

  for (auto &Node : utils::makeRange(HostGraph.nodesBeg(), 
                                     HostGraph.nodesEnd())) {
    auto [NodeIdx, _] = GetIndexes(Node.get(), Node.get());
    fillAdjMatrix(NodeIdx, NodeIdx, CostMatrices.size());
    CostMatrices.emplace_back(Node->getCostVector());
  }

  NumOfCosts = CostMatrices.size();
  Costs = copyVectorToCuda(CostMatrices);
}

__host__
void Graph::free() {
  AdjMatrix.free();
  for (auto &DevMatr : CostMatrices)
    DevMatr.free();
  CUDA_CHECK(cudaFree(Costs));
}

__host__
size_t Graph::getNumOfCostCombinations() const {
  auto NumOfVariants = 1ull;
  assert(HostAdjMatrix.h() == HostAdjMatrix.w());
  for (size_t NodeIdx = 0; NodeIdx < HostAdjMatrix.h(); ++NodeIdx) {
    auto CostIdx = HostAdjMatrix[NodeIdx][NodeIdx];
    if (CostIdx != NoEdge) {
      assert(CostIdx < CostMatrices.size());
      assert(CostMatrices[CostIdx].w() == 1);
      NumOfVariants *= CostMatrices[CostIdx].h();
    }
  }
  assert(NumOfVariants != 0);
  return NumOfVariants;
}

} // namespace device