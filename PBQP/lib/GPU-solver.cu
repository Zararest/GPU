#include "GPU-solver.cu.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

namespace PBQP {

namespace {

__global__
void __calcCosts(device::Graph Graph, Graph::Cost_t *AllCosts, unsigned NumOfCombinations) {
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

  if (GlobalId < NumOfCombinations)
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
    auto MinElemIt = thrust::min_element(Costs.begin(), Costs.end());
    assert(MinElemIt != Costs.end());
    auto MinElemIdx = std::distance(Costs.begin(), MinElemIt);
    auto Solution = getSolutionByIndex(Graph, MinElemIdx);
    Solution.addFinalCost(*MinElemIt);
    return Solution;
  }

  Solution getOptimalSolution(device::Graph &Graph) {
    auto NumOfCombinations = Graph.getNumOfCostCombinations();
    thrust::device_vector<Graph::Cost_t> 
      AllCosts(NumOfCombinations, Graph::InfCost);
    dim3 ThrBlockDim{BlockSize};
    dim3 BlockGridDim{utils::ceilDiv(NumOfCombinations, ThrBlockDim.x)};
    __calcCosts<<<BlockGridDim, ThrBlockDim>>>
      (Graph, thrust::raw_pointer_cast(AllCosts.data()), NumOfCombinations);
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

void GPUSolver::PassManager::addPass(Pass_t Pass, std::string Name) {
  if (Name == "")
    Name = "Pass " + std::to_string(PassPtrToName.size());
  PassPtrToName[Pass.get()] = Name;
  Passes.emplace_back(std::move(Pass));
}

Solution GPUSolver::PassManager::run(Graph Graph) {
  auto Res = Pass::Res_t{};
  for (auto &Pass : Passes) {
    auto Start = std::chrono::steady_clock::now();
    Res = Pass->run(Graph, std::move(Res));
    auto End = std::chrono::steady_clock::now();
    PassPtrToDuration[Pass.get()] = utils::to_milliseconds(End - Start);
  }
  auto *SolutionPtr = dynamic_cast<FinalSolution *>(Res.get());
  if (!SolutionPtr)
    utils::reportFatalError("Invalid final pass");
  
  return SolutionPtr->getFinalSolution(std::move(Graph));
}

GPUSolver::PassManager::Profile_t
GPUSolver::PassManager::getProfileInfo() const {
  auto Res = std::vector<std::pair<std::string, size_t>>{};
  std::transform(Passes.begin(), Passes.end(),
                 std::back_inserter(Res),
                 [&](const Pass_t &Pass) {
                  auto PassPtr = Pass.get();
                  auto NameIt = PassPtrToName.find(PassPtr);
                  assert(NameIt != PassPtrToName.end());
                  auto DurationIt = PassPtrToDuration.find(PassPtr);
                  assert(DurationIt != PassPtrToDuration.end());
                  return std::pair<std::string, size_t>{NameIt->second, 
                                                        DurationIt->second};
                 });
  return Res;
}

Solution GPUSolver::solve(Graph Task) {
  this->addPasses(PM);
  return PM.run(std::move(Task));           
}

void GPUFullSearch::addPasses(PassManager &PM) {
  PM.addPass(Pass_t{new GraphLoader});
  PM.addPass(Pass_t{new FullSearchImpl}, "GPU full search");
  PM.addPass(Pass_t{new GraphDeleter});
}

} // namespace PBQP