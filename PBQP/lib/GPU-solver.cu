#include "GPU-solver.cu.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

namespace PBQP {

namespace {

__global__
void __calcCosts(device::Graph Graph, Graph::Cost_t *AllCosts) {
  // This should be stored in local memory
  auto &AdjMatrix = Graph.getAdjMatrix();
  auto GlobalId = blockIdx.x * blockDim.x + threadIdx.x;
  auto LhsChoicesLeft = GlobalId;
  auto RhsChoicesLeft = LhsChoicesLeft;
  auto NumOfNodes = AdjMatrix.h();
  auto Cost = Graph::Cost_t{0};
  for (unsigned LhsNodeIdx = 0; LhsNodeIdx < NumOfNodes; ++LhsNodeIdx) {
    auto LhsCostIdx = AdjMatrix[LhsNodeIdx][LhsNodeIdx];
    auto &LhsCostVect = Graph.getCostMatrix(LhsCostIdx);
    assert(LhsCostVect.w() == 1);
    auto LhsChoice = LhsChoicesLeft % LhsCostVect.h();
    LhsChoicesLeft /= LhsCostVect.h();
    Cost += LhsCostVect[LhsChoice][0];
    RhsChoicesLeft = LhsChoicesLeft;
    for (unsigned RhsNodeIdx = LhsNodeIdx + 1; RhsNodeIdx < NumOfNodes; ++RhsNodeIdx) {
      auto AdjCostIdx = AdjMatrix[LhsNodeIdx][RhsNodeIdx];
      auto RhsCostSize = Graph.getCostMatrix(AdjMatrix[RhsNodeIdx][RhsNodeIdx]).h();
      if (AdjCostIdx >= 0) {
        auto RhsChoice = RhsChoicesLeft % RhsCostSize;
        Cost += Graph.getCostMatrix(AdjCostIdx)[LhsChoice][RhsChoice];
      }
      RhsChoicesLeft /= RhsCostSize;
    }
  }
  // FXME
  if (GlobalId < 9)
    AllCosts[GlobalId] = Cost;
}

// Class for passing device::Graph through PassManager
struct GPUGraph final : public GPUSolver::Pass::Result {
  device::Graph Graph;

  GPUGraph(const PBQP::Graph &HostGraph) : Graph{HostGraph} {}
};

struct GPUResult final : public GPUSolver::Pass::Result {
  device::Graph Graph;
  Solution Sol;

  GPUResult(device::Graph Graph, Solution Sol = Solution{}) : Graph{Graph},
                                                              Sol{std::move(Sol)} {}
};

// Pass which creates device::Graph and passes it further
struct GraphLoader final : public GPUSolver::Pass {
  Res_t run(const Graph &Graph, Res_t PrevResult) override {
    return Res_t{new GPUGraph(Graph)};
  }
};

// Pass which finds optimal solution with full search 
//  on GPU graph received from previous pass
class FullSearchImpl final : public GPUSolver::Pass {
  static constexpr size_t BlockSize = 32;

  Solution getSolutionByIndex(device::Graph &Graph, unsigned SelectedVariant) {
    auto Res = Solution{};
    for (unsigned NodeIdx = 0; NodeIdx < Graph.size(); ++NodeIdx) {
      auto NodeCostSize = Graph.getNodeCostSize(NodeIdx);
      Res.addSelection(NodeIdx, SelectedVariant % NodeCostSize);
      SelectedVariant /= NodeCostSize;
    }
    return Res;
  }

  Solution findSolutionWithMinCost(device::Graph &Graph, 
                                   thrust::device_vector<Graph::Cost_t> Costs) {
    //thrust::copy(Costs.begin(), Costs.end(), std::ostream_iterator<float>(std::cout, " "));
    auto MinElemIt = thrust::min_element(Costs.begin(), Costs.end());
    assert(MinElemIt != Costs.end());
    auto MinElemIdx = std::distance(Costs.begin(), MinElemIt);
    return getSolutionByIndex(Graph, MinElemIdx);
  }

  Solution getOptimalSolution(device::Graph &Graph) {
    auto NumOfCombinations = Graph.getNumOfCostCombinations();
    thrust::device_vector<Graph::Cost_t> 
      AllCosts(NumOfCombinations, Graph::InfCost);
    dim3 ThrBlockDim{BlockSize};
    dim3 BlockGridDim{utils::ceilDiv(NumOfCombinations, ThrBlockDim.x)};
    __calcCosts<<<BlockGridDim, ThrBlockDim>>>
      (Graph, thrust::raw_pointer_cast(AllCosts.data()));
    cudaDeviceSynchronize();
    utils::checkKernelsExec();
    return findSolutionWithMinCost(Graph, std::move(AllCosts));
  }

public:
  Res_t run(const Graph &Graph, Res_t PrevResult) override {
    auto *GPUGraphPtr = dynamic_cast<GPUGraph *>(PrevResult.get());
    if (!GPUGraphPtr)
      utils::reportFatalError("Graph hasn't been loaded to GPU");
    return Res_t{new GPUResult(GPUGraphPtr->Graph, 
                               getOptimalSolution(GPUGraphPtr->Graph))};
  }
};

// Final pass which frees GPU memory
struct GraphDeleter final : public GPUSolver::FinalPass {
  Solution getSolution(const Graph &Graph, Res_t PrevResult) override {
    auto *GPURes = dynamic_cast<GPUResult *>(PrevResult.get());
    if (!GPURes)
      utils::reportFatalError("There is no GPU solvers in PM");
    GPURes->Graph.free();
    return std::move(GPURes->Sol);
  }
};

} // anonymous namespace

GPUSolver::Pass::Result::~Result() {}

void GPUSolver::PassManager::addPass(Pass_t Pass) {
  Passes.emplace_back(std::move(Pass));
}

Solution GPUSolver::PassManager::run(Graph Graph) {
  auto Res = Pass::Res_t{};
  for (auto &Pass : Passes) {
    Res = Pass->run(Graph, std::move(Res));
  }
  auto *SolutionPtr = dynamic_cast<FinalSolution *>(Res.get());
  if (!SolutionPtr)
    utils::reportFatalError("Invalid final pass");
  
  return SolutionPtr->getFinalSolution(std::move(Graph));
}

Solution GPUSolver::solve(Graph Task) {
  auto PM = PassManager{};
  this->addPasses(PM);
  return PM.run(std::move(Task));           
}

void GPUFullSearch::addPasses(PassManager &PM) {
  PM.addPass(Pass_t{new GraphLoader});
  PM.addPass(Pass_t{new FullSearchImpl});
  PM.addPass(Pass_t{new GraphDeleter});
}

} // namespace PBQP