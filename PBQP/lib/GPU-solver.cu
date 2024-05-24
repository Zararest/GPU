#include "GPU-solver.cu.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>

namespace PBQP {

namespace {

__device__
void __fillChoices(device::Graph &Graph, unsigned char *Choices, unsigned GlobalId) {
  // We assume that task with node size more than 255 is too hard
  constexpr auto MaxNodeSize = 255;
  auto NumOfNodes = Graph.getAdjMatrix().h();
  auto &AdjMatrix = Graph.getAdjMatrix();
  for (unsigned NodeIdx = 0; NodeIdx < NumOfNodes; ++NodeIdx) {
    auto CostIdx = AdjMatrix[NodeIdx][NodeIdx];
    auto &CostVect = Graph.getCostMatrix(CostIdx);
    assert(CostVect.w() == 1);
    assert(CostVect.h() <= MaxNodeSize);
    Choices[NodeIdx] = GlobalId % CostVect.h();
    GlobalId /= CostVect.h();
  }
}

__device__
Graph::Cost_t __calcMatrixesCost(device::Graph &Graph, unsigned LhsIdx, 
                                 unsigned char *Choices) {
  auto &AdjMatrix = Graph.getAdjMatrix();
  auto NumOfNodes = AdjMatrix.h();
  auto Cost = Graph::Cost_t{0};
  auto LhsChoice = Choices[LhsIdx];
  for (unsigned RhsIdx = 0; RhsIdx < NumOfNodes; ++RhsIdx) {
    if (LhsIdx == RhsIdx)
      continue;
    auto AdjCostIdx = AdjMatrix[LhsIdx][RhsIdx];
    if (AdjCostIdx >= 0) {
      auto RhsChoice = Choices[RhsIdx];
      Cost += Graph.getCostMatrix(AdjCostIdx)[LhsChoice][RhsChoice];
    }
  }
  return Cost;
}

__global__
void __calcCosts(device::Graph Graph, Graph::Cost_t *AllCosts, unsigned NumOfCombinations) {
  // We assume that task on more than 32 nodes is unsolvable, 
  //  because 2^32 at least alot
  constexpr auto MaxNumberOfNodes = 32u;
  unsigned char Choices[MaxNumberOfNodes];
  auto GlobalId = blockIdx.x * blockDim.x + threadIdx.x;
  __fillChoices(Graph, Choices, GlobalId);
  auto Cost = Graph::Cost_t{0};
  auto &AdjMatrix = Graph.getAdjMatrix();
  auto NumOfNodes = AdjMatrix.h();
  for (unsigned LhsIdx = 0; LhsIdx < NumOfNodes; ++LhsIdx) {
    Cost += __calcMatrixesCost(Graph, LhsIdx, Choices);
    auto LhsVectorCostIdx = AdjMatrix[LhsIdx][LhsIdx];
    assert(LhsVectorCostIdx >= 0);
    auto LhsChoice = Choices[LhsIdx];
    Cost += Graph.getCostMatrix(LhsVectorCostIdx)[LhsChoice][0];
  }

  if (GlobalId < NumOfCombinations)
    AllCosts[GlobalId] = Cost;
}

// Returns number of selection for node 
//  and removes cost vector of a given node from AdjMatrix
__device__
int reduceNodeR0(device::Graph &Graph, unsigned NodeId) {
  auto CostVectorId = Graph.getAdjMatrix()[NodeId][NodeId];
  assert(CostVectorId >= 0);
  auto &CostVector = Graph.getCostMatrix(CostVectorId);
  auto CostVectorSize = CostVector.h();
  assert(CostVector.w() == 1);
  auto Solution = 0;
  auto MinCost = CostVector[0][0];
  for (unsigned i = 1; i < CostVectorSize; ++i) {
    auto NewCost = CostVector[i][0];
    if (NewCost < MinCost) {
      Solution = i;
      MinCost = NewCost;
    }
  }
  Graph.getAdjMatrix()[NodeId][NodeId] = -1;
  return Solution;
}

__global__
void __R0Reduction(device::Graph Graph, char *NodesToReduce, int *SelectionForNodes) {
  auto GlobalId = blockIdx.x * blockDim.x + threadIdx.x;
  auto NumOfNodes = Graph.getAdjMatrix().w();
  if (GlobalId < NumOfNodes && NodesToReduce[GlobalId] == 1)
    SelectionForNodes[GlobalId] = reduceNodeR0(Graph, GlobalId);
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

struct FinalMock final : public GPUSolver::FinalPass {
  Solution getSolution(const Graph &Graph, Res_t PrevResult) override {
    return Solution{};
  }
};

class LoopCondition : public GPUSolver::Pass::Result {
protected:  
  bool Condition = false;

public:
  bool getCondition() const { return Condition; }
  void setCondition(bool NewCond) { Condition = NewCond; }
};

struct LoopConditionHandler final : public GPUSolver::Condition {
  bool check(GPUSolver::Pass::Res_t &PrevResult) override {
    auto *ResPtr = dynamic_cast<LoopCondition *>(PrevResult.get());
    if (!ResPtr)
      utils::reportFatalError("Loop header accepts only LoopCondition class");
    return ResPtr->getCondition();
  }
};

class LoopCounter : public LoopCondition {
protected:
  size_t NumOfIterations;
  size_t CurIteration = 0;

  void checkCondition() {
    Condition = CurIteration < NumOfIterations;
  }

public:
  LoopCounter(size_t NumOfIterations) : NumOfIterations{NumOfIterations} {
    checkCondition();
  } 

  void inc() {
    CurIteration++;
    checkCondition();
  }
};

struct Counter final : public GPUSolver::Pass {
  Res_t run(const Graph &Graph, Res_t PrevResult) override {
    auto *ConterResPtr = dynamic_cast<LoopCounter *>(PrevResult.get());
    if (!ConterResPtr)
      utils::reportFatalError("Counter accepts only LoopCounter result");
    ConterResPtr->inc();
    return PrevResult;
  }
};

struct CounterInit final : public GPUSolver::Pass {
  Res_t run(const Graph &Graph, Res_t PrevResult) override {
    constexpr auto IterNum = 10;
    return Res_t{new LoopCounter(IterNum)};
  }
};

std::pair<std::vector<char>, bool> getNodesToReduceR0(const Graph &Graph) {
  auto NodesToReduce = std::vector<char>(Graph.size(), 0);
  auto HasNodeToReduce = false;
  //TODO: make indexed range
  auto NodeIdx = 0ul;
  for (auto &Node : utils::makeRange(Graph.nodesBeg(), Graph.nodesEnd())) {
    if (Node->order() == 0) {
      NodesToReduce[NodeIdx] = 1;
      HasNodeToReduce = true;
    }
    NodeIdx++;
  }
  return {NodesToReduce, HasNodeToReduce};
}

void commitReductionToHost(thrust::host_vector<int> SelectionForNodes, 
                           device::Graph &GraphDevice, Solution &Sol) {
  for (size_t NodeIdx = 0; NodeIdx < SelectionForNodes.size(); ++NodeIdx) {
    auto Selection = SelectionForNodes[NodeIdx];
    if (Selection >= 0) {
      Sol.addSelection(NodeIdx, Selection);
      GraphDevice.makeCostUnreachable(NodeIdx, NodeIdx);
    }
  }
}

bool performR0Reduction(device::Graph &GraphDevice, const Graph &Graph, 
                        Solution &Sol, unsigned BlockSize) {
  auto [NodesToReduce, HasNodeToReduce] = getNodesToReduceR0(Graph);
  auto NodesToReduceDevice = 
    thrust::device_vector<char>(NodesToReduce.begin(), NodesToReduce.end());
  auto SelectionForNodes = thrust::device_vector<int>(NodesToReduce.size(), -1);
  dim3 ThrBlockDim{BlockSize};
  dim3 BlockGridDim{utils::ceilDiv(Graph.size(), ThrBlockDim.x)};
  __R0Reduction<<<BlockGridDim, ThrBlockDim>>>
    (GraphDevice, thrust::raw_pointer_cast(NodesToReduce.data()), 
                  thrust::raw_pointer_cast(SelectionForNodes.data()));
  commitReductionToHost(std::move(SelectionForNodes), GraphDevice, Sol);
  return HasNodeToReduce;
}

} // anonymous namespace

GPUSolver::Pass::Result::~Result() {}

void GPUSolver::PassManager::addPass(Pass_t Pass, std::string Name) {
  if (Name == "")
    Name = "Pass " + std::to_string(PassPtrToName.size());
  PassPtrToName[Pass.get()] = Name;
  Stages.emplace_back(std::move(Pass));
}

void GPUSolver::PassManager::addLoopStart(Condition_t Cond) {
  LoopHeaderIdxes.push_back(Stages.size());
  Stages.emplace_back(LoopHeader{std::move(Cond)});
}

void GPUSolver::PassManager::addLoopEnd() {
  if (LoopHeaderIdxes.empty())
    utils::reportFatalError("Loop header hasn't been specified");
  auto PrevHeader = LoopHeaderIdxes.back();
  LoopHeaderIdxes.pop_back();
  auto &HeaderStage = Stages[PrevHeader];
  assert(std::holds_alternative<LoopHeader>(HeaderStage));
  std::get<LoopHeader>(HeaderStage).EndIdx = Stages.size();
  Stages.emplace_back(LoopEnd{PrevHeader});
}

GPUSolver::Res_t 
GPUSolver::PassManager::runPass(Pass_t &Pass, Res_t PrevRes, Graph &Graph) {
  auto Start = std::chrono::steady_clock::now();
  auto Res = Pass->run(Graph, std::move(PrevRes));
  auto End = std::chrono::steady_clock::now();
  PassPtrToDuration[Pass.get()] += utils::to_milliseconds(End - Start);
  return Res;
}

size_t GPUSolver::PassManager::getNextIdx(LoopHeader &Header, 
                                          Res_t &Res, size_t CurIdx) {
  if (Header.Cond->check(Res))
    return CurIdx + 1;
  return Header.EndIdx + 1;
}

Solution GPUSolver::PassManager::run(Graph Graph) {
  if (!LoopHeaderIdxes.empty())
    utils::reportFatalError("Theree is a loop header without end");
  auto Res = Pass::Res_t{};
  for (size_t CurStageIdx = 0; CurStageIdx < Stages.size();) {
    auto &CurStage = Stages[CurStageIdx];
    if (std::holds_alternative<Pass_t>(CurStage)) {
      Res = runPass(std::get<Pass_t>(CurStage), std::move(Res), Graph);
      CurStageIdx++;
      continue;
    }
    if (std::holds_alternative<LoopHeader>(CurStage)) {
      CurStageIdx = getNextIdx(std::get<LoopHeader>(CurStage), Res, CurStageIdx);
      continue;
    }
    if (std::holds_alternative<LoopEnd>(CurStage)) {
      auto &HeaderIdx = std::get<LoopEnd>(CurStage).HeaderIdx;
      auto &Header = Stages[HeaderIdx];
      assert(std::holds_alternative<LoopHeader>(Header)); 
      // We've jumped to the header
      CurStageIdx = getNextIdx(std::get<LoopHeader>(Header), Res, HeaderIdx);
      continue;
    }
  }

  auto *SolutionPtr = dynamic_cast<FinalSolution *>(Res.get());
  if (!SolutionPtr)
    utils::reportFatalError("Invalid final pass");
  
  return SolutionPtr->getFinalSolution(std::move(Graph));
}

GPUSolver::PassManager::Profile_t
GPUSolver::PassManager::getProfileInfo() const {
  auto Res = std::vector<std::pair<std::string, size_t>>{};
  std::transform(Stages.begin(), Stages.end(),
                 std::back_inserter(Res),
                [&](const auto &Stage) {
                  if (std::holds_alternative<LoopHeader>(Stage))
                    return std::pair<std::string, size_t>{"Loop header", 0};
                  if (std::holds_alternative<LoopEnd>(Stage))
                    return std::pair<std::string, size_t>{"Loop end", 0};
                  auto PassPtr = std::get<Pass_t>(Stage).get();
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

// State of loop determines the condition on its end.
// This class contains metadata for graph processing
template <typename T>
struct LoopState : public LoopCondition {
  using Metadata_t = std::unique_ptr<T>;
private:
  Metadata_t Metadata;

public:
  LoopState(Metadata_t Metadata) : Metadata{std::move(Metadata)} {}

  T &getMetadata() {
    if (!Metadata)
      utils::reportFatalError("There is no metadata");
    return *Metadata;
  }
};

class GPUGraphData final {
  device::Graph Graph;
  bool GraphChanged = false;
  Solution Sol;

public:
  GPUGraphData(const PBQP::Graph &HostGraph) : Graph{HostGraph} {}

  device::Graph &getDeviceGraph() { return Graph; }
  void graphHasBeenChanged(bool IsChanged) { GraphChanged = IsChanged; }
  bool isGraphChanged() const { return GraphChanged; }
  Solution &getSolution() { return Sol; }
};

struct InitStatePass final : public GPUSolver::Pass {
  Res_t run(const Graph &Graph, Res_t PrevResult) override {
    auto Metadata = LoopState<GPUGraphData>::Metadata_t{new GPUGraphData(Graph)};
    auto NewLoopState = new LoopState<GPUGraphData>(std::move(Metadata));
    NewLoopState->setCondition(/*NewCond*/ true);
    return Res_t{NewLoopState};
  }
};

struct GraphChangeChecker final : public GPUSolver::Pass {
  Res_t run(const Graph &Graph, Res_t PrevResult) override {
    auto *ResPtr = dynamic_cast<LoopState<GPUGraphData> *>(PrevResult.get());
    if (!ResPtr)
      utils::reportFatalError("Invalid loop state");
    auto &Metadata = ResPtr->getMetadata();
    ResPtr->setCondition(Metadata.isGraphChanged());
    return PrevResult;
  }
};

GPUSolver::Res_t 
HeuristicSolver::R0Reduction::run(const Graph &Graph, 
                                  GPUSolver::Res_t PrevResult) {
  auto *ResPtr = dynamic_cast<LoopState<GPUGraphData> *>(PrevResult.get());
  if (!ResPtr)
    utils::reportFatalError("Invalid loop state in R0 reduction");
  auto &Metadata = ResPtr->getMetadata();
  bool Changed = performR0Reduction(Metadata.getDeviceGraph(), 
                                    Graph,
                                    Metadata.getSolution(),
                                    BlockSize) || 
                  Metadata.isGraphChanged();
  ResPtr->setCondition(Changed);
  return PrevResult;
}

void HeuristicSolver::addPasses(PassManager &PM) {
  PM.addPass(Pass_t{new InitStatePass});
  PM.addLoopStart(Condition_t{new LoopConditionHandler});
    PM.addPass(Pass_t{new R0Reduction});
    PM.addPass(Pass_t{new R0Reduction});
    PM.addPass(Pass_t{new GraphChangeChecker});
  PM.addLoopEnd();
  PM.addPass(Pass_t{new FinalMock});
}

} // namespace PBQP