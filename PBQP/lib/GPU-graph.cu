#include "GPU-graph.h"

#include <map>

namespace {
template <typename T> T *copyVectorToCuda(const std::vector<T> &Vect) {
  T *CostsPtr = nullptr;
  CUDA_CHECK(cudaMalloc((void **)&CostsPtr, Vect.size() * sizeof(T)));
  for (size_t Idx = 0; Idx < Vect.size(); ++Idx)
    CUDA_CHECK(cudaMemcpy(&CostsPtr[Idx], &Vect[Idx], sizeof(T),
                          cudaMemcpyHostToDevice));
  return CostsPtr;
}

} // anonymous namespace

namespace device {

__host__ Graph::Graph(const PBQP::Graph &HostGraph) {
  for (size_t NodeIdx = 0; NodeIdx < HostGraph.size(); NodeIdx++)
    Translator.addNodes(NodeIdx, NodeIdx);

  auto NodeAddrToIdx = std::map<PBQP::Graph::Node *, size_t>{};
  auto Idx = 0ull;
  for (auto &NodePtr :
       utils::makeRange(HostGraph.nodesBeg(), HostGraph.nodesEnd())) {
    NodeAddrToIdx[NodePtr.get()] = Idx;
    ++Idx;
  }

  auto GetIndexes = [&NodeAddrToIdx](auto *Lhs, auto *Rhs) {
    assert(NodeAddrToIdx.find(Lhs) != NodeAddrToIdx.end());
    assert(NodeAddrToIdx.find(Rhs) != NodeAddrToIdx.end());
    return std::make_pair(NodeAddrToIdx[Lhs], NodeAddrToIdx[Rhs]);
  };

  HostAdjMatrix = host::Matrix<Index_t>{HostGraph.size(), HostGraph.size()};
  std::fill(HostAdjMatrix.begin(), HostAdjMatrix.end(), NoEdge);
  for (auto &Edge :
       utils::makeRange(HostGraph.edgesBeg(), HostGraph.edgesEnd())) {
    auto [Lhs, Rhs] = Edge->getNodes();
    auto [LhsIdx, RhsIdx] = GetIndexes(Lhs, Rhs);
    HostAdjMatrix[LhsIdx][RhsIdx] = CostMatrices.size();
    CostMatrices.emplace_back(Edge->getCostMatrix());
  }

  for (auto &Node :
       utils::makeRange(HostGraph.nodesBeg(), HostGraph.nodesEnd())) {
    auto [NodeIdx, _] = GetIndexes(Node.get(), Node.get());
    HostAdjMatrix[NodeIdx][NodeIdx] = CostMatrices.size();
    CostMatrices.emplace_back(Node->getCostVector());
  }

  NumOfCosts = CostMatrices.size();
  Costs = copyVectorToCuda(CostMatrices);
  AdjMatrix = device::Matrix<Index_t>{HostAdjMatrix};
}

__host__ void Graph::free() {
  AdjMatrix.free();
  for (auto &DevMatr : CostMatrices)
    DevMatr.free();
  CUDA_CHECK(cudaFree(Costs));
}

__host__ size_t Graph::getNumOfCostCombinations() const {
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

__host__ void Graph::removeUnreachableCosts() {
  for (auto Unreachable : UnreachableCosts) {
    DEBUG_EXPR(std::cout << "Removing unreachable: " << 
               Unreachable << "\n");
    // FIXME with this cleanup pass is not working
    CostMatrices[Unreachable].free();
    CostMatrices[Unreachable] = device::Matrix<Cost_t>{};
  }
  UnreachableCosts.clear();
}

__host__ void Graph::updateTranslator() {
  auto NewTranslator = NodesTranslator{};
  for (size_t CurDeviceIdx = 0; CurDeviceIdx < HostAdjMatrix.w();
       CurDeviceIdx++) {
    auto HostNodeIdx = getHostNode(CurDeviceIdx);
    if (HostNodeIdx) {
      auto NewIdx = NewTranslator.getMaxDeviceIdx()
                        ? *NewTranslator.getMaxDeviceIdx() + 1
                        : 0;
      NewTranslator.addNodes(NewIdx, *HostNodeIdx);
    }
  }
  Translator = NewTranslator;
}

__host__ void Graph::updateHostAdjMatrix() {
  auto NumOfUnresolvedNodes =
      Translator.getMaxDeviceIdx() ? *Translator.getMaxDeviceIdx() + 1 : 0;
  assert(HostAdjMatrix.h() == HostAdjMatrix.w());
  auto MatrixSize = HostAdjMatrix.h();
  auto NewAdjMatrixValue = std::vector<Index_t>{};
  for (size_t i = 0; i < MatrixSize; ++i)
    if (nodeIsUnresolved(i))
      for (size_t j = 0; j < MatrixSize; ++j)
        if (nodeIsUnresolved(j))
          NewAdjMatrixValue.push_back(HostAdjMatrix[i][j]);
  auto NewAdjMatrix =
      host::Matrix<Index_t>(NewAdjMatrixValue.begin(), NewAdjMatrixValue.end(),
                            NumOfUnresolvedNodes, NumOfUnresolvedNodes);
  HostAdjMatrix = std::move(NewAdjMatrix);
}
__host__ void Graph::removeUnreachableNodes() {
  removeUnreachableCosts();
  updateTranslator();
  updateHostAdjMatrix();

  AdjMatrix.free();
  AdjMatrix = device::Matrix<Index_t>{HostAdjMatrix};
  assert(checkAdjMatricesCoherence());
}

__host__ bool Graph::checkAdjMatricesCoherence() const {
  auto CurDeviceAdjMatrix = AdjMatrix.getHostMatrix();
  return checkHostMatrix() &&
         std::equal(CurDeviceAdjMatrix.begin(), CurDeviceAdjMatrix.end(),
                    HostAdjMatrix.begin(), HostAdjMatrix.end());
}

thrust::host_vector<unsigned> Graph::getNeighbours(unsigned NodeIdx) {
  auto Res = thrust::host_vector<unsigned>{};
  auto NumOfNodes = size();
  constexpr auto NoEdge = -1;

  for (size_t i = 0; i < NumOfNodes; ++i) {
    if (i != NodeIdx && 
        (HostAdjMatrix[i][NodeIdx] != NoEdge ||
         HostAdjMatrix[NodeIdx][i] != NoEdge))
      Res.push_back(i);
  }
  return Res;
}

} // namespace device