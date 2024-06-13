#include "GPU-solver.cu.h"

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <unordered_set>

namespace PBQP {

namespace {

__device__ void __fillChoices(device::Graph &Graph, unsigned char *Choices,
                              unsigned GlobalId) {
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

__device__ Graph::Cost_t __calcMatrixesCost(device::Graph &Graph,
                                            unsigned LhsIdx,
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

__global__ void __calcCosts(device::Graph Graph, Graph::Cost_t *AllCosts,
                            unsigned NumOfCombinations) {
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
__device__ int __reduceNodeR0(device::Graph &Graph, unsigned NodeId) {
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

__global__ void __R0Reduction(device::Graph Graph, char *NodesToReduce,
                              int *SelectionForNodes) {
  auto GlobalId = blockIdx.x * blockDim.x + threadIdx.x;
  auto NumOfNodes = Graph.getAdjMatrix().w();
  if (GlobalId < NumOfNodes && NodesToReduce[GlobalId] == 1)
    SelectionForNodes[GlobalId] = __reduceNodeR0(Graph, GlobalId);
}

// Class for passing device::Graph through PassManager
struct GPUGraph final : public GPUSolver::Pass::Result {
  device::Graph Graph;

  GPUGraph(const PBQP::Graph &HostGraph) : Graph{HostGraph} {}
};

struct GPUResult final : public GPUSolver::Pass::Result {
  device::Graph Graph;
  Solution Sol;

  GPUResult(device::Graph Graph, Solution Sol = Solution{})
      : Graph{std::move(Graph)}, Sol{std::move(Sol)} {}
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
    thrust::device_vector<Graph::Cost_t> AllCosts(NumOfCombinations,
                                                  Graph::InfCost);
    dim3 ThrBlockDim{BlockSize};
    dim3 BlockGridDim{utils::ceilDiv(NumOfCombinations, ThrBlockDim.x)};
    __calcCosts<<<BlockGridDim, ThrBlockDim>>>(
        Graph, thrust::raw_pointer_cast(AllCosts.data()), NumOfCombinations);
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

  void checkCondition() { Condition = CurIteration < NumOfIterations; }

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

__global__ void __getNodesToReduceR0(device::Graph Graph, char *NodesToReduce) {
  auto &AdjMatrix = Graph.getAdjMatrix();
  auto GraphSize = AdjMatrix.h();
  auto GlobalId = blockIdx.x * blockDim.x + threadIdx.x;

  if (GlobalId >= GraphSize)
    return;

  auto NumOfNeighb = 0u;
  for (unsigned i = 0; i < GraphSize; ++i) {
    if (i == GlobalId)
      continue;
    NumOfNeighb += AdjMatrix[i][GlobalId] != -1;
    NumOfNeighb += AdjMatrix[GlobalId][i] != -1;
  }
  NodesToReduce[GlobalId] = NumOfNeighb == 0;
}

// returns vector with 0 and 1 with the size of the initial graph
std::pair<thrust::device_vector<char>, size_t>
getNodesToReduceR0(device::Graph &CurGraph, unsigned BlockSize) {
  auto NumOfNodes = CurGraph.size();
  auto NodesToReduceDevice = thrust::device_vector<char>(NumOfNodes);
  dim3 ThrBlockDim{BlockSize};
  dim3 BlockGridDim{utils::ceilDiv(NumOfNodes, ThrBlockDim.x)};
  __getNodesToReduceR0<<<BlockGridDim, ThrBlockDim>>>(
      CurGraph, thrust::raw_pointer_cast(NodesToReduceDevice.data()));
  auto NumOfNodesToReduce =
      thrust::reduce(NodesToReduceDevice.begin(), NodesToReduceDevice.end());
  return {NodesToReduceDevice, NumOfNodesToReduce};
}

void commitR0ReductionToHost(thrust::host_vector<int> SelectionForNodes,
                             device::Graph &GraphDevice, Solution &Sol) {
  for (size_t NodeIdx = 0; NodeIdx < SelectionForNodes.size(); ++NodeIdx) {
    auto Selection = SelectionForNodes[NodeIdx];
    if (Selection >= 0) {
      auto HostIdx = GraphDevice.getHostNode(NodeIdx);
      assert(HostIdx && "Node already has been resolved");
      Sol.addSelection(*HostIdx, Selection);
      GraphDevice.makeCostUnreachable(NodeIdx, NodeIdx);
    }
  }
}

bool performR0Reduction(device::Graph &GraphDevice, const Graph &Graph,
                        Solution &Sol, unsigned BlockSize) {
  DEBUG_EXPR(std::cout << "\n---------------------\n");
  DEBUG_EXPR(std::cout << "Performing R0 reduction\n");
  auto [NodesToReduceDevice, NumOfNodesToReduce] =
      getNodesToReduceR0(GraphDevice, BlockSize);
  DEBUG_EXPR(std::cout << "Found " << NumOfNodesToReduce
                       << " nodes\n");
  if (NumOfNodesToReduce == 0)
    return false;

  auto SelectionForNodes =
      thrust::device_vector<int>(NodesToReduceDevice.size(), -1);
  dim3 ThrBlockDim{BlockSize};
  dim3 BlockGridDim{utils::ceilDiv(Graph.size(), ThrBlockDim.x)};
  __R0Reduction<<<BlockGridDim, ThrBlockDim>>>(
      GraphDevice, thrust::raw_pointer_cast(NodesToReduceDevice.data()),
      thrust::raw_pointer_cast(SelectionForNodes.data()));
  commitR0ReductionToHost(std::move(SelectionForNodes), GraphDevice, Sol);
  return NumOfNodesToReduce != 0;
}

class NodesToReduceR1 {
  int NodeWithSingleNeighb = -1;
  int Heighbour = -1;
  utils::Pair<int, int> MatrixPos{-1, -1};

public:
  __host__ __device__ void addSingleNode(int NewNode) {
    NodeWithSingleNeighb = NewNode;
  }

  __host__ __device__ unsigned getSingleNode() const {
    assert(NodeWithSingleNeighb >= 0);
    return static_cast<unsigned>(NodeWithSingleNeighb);
  }

  __host__ __device__ void addNeighbour(int NewNode) { Heighbour = NewNode; }

  __host__ __device__ unsigned getNeighbour() const {
    assert(Heighbour >= 0);
    return static_cast<unsigned>(Heighbour);
  }

  __host__ __device__ void addAdjMatrixPos(unsigned Y, unsigned X) {
    MatrixPos.First = Y;
    MatrixPos.Second = X;
  }

  __host__ __device__ bool canBeReduced() const {
    return NodeWithSingleNeighb != -1 && Heighbour != -1;
  }

  __host__ __device__ utils::Pair<unsigned, unsigned> getPosOfCostMatix() const {
    assert(MatrixPos.First != -1 && MatrixPos.Second != -1);
    return utils::Pair<unsigned, unsigned>{
        static_cast<unsigned>(MatrixPos.First),
        static_cast<unsigned>(MatrixPos.Second)};
  }

  // There are 2 variants of nodes:
  //  1) Single -> Neighb -> ...
  //                  |
  //                  V
  //                 ...
  //  2) Single <- Neighb -> ...
  //                  |
  //                  V
  //                 ...
  // In order to make abstraction over these 2 cases,
  //  this class translates selcetions in terms of Single/Neighb
  //  into real matrix pos.
  template <typename T>
  __device__ auto getCostForSelections(device::Matrix<T> &CostMatrix,
                                       unsigned SingleNodeSelection,
                                       unsigned NeighbourSelection) {
    if (NodeWithSingleNeighb == MatrixPos.First)
      return CostMatrix[SingleNodeSelection][NeighbourSelection];
    return CostMatrix[NeighbourSelection][SingleNodeSelection];
  }
};

__device__ utils::Pair<unsigned, Graph::Cost_t> 
__getBestDependentSelection(unsigned NeighbSelection, 
                            device::Matrix<Graph::Cost_t> &CostMatrix,
                            device::Matrix<Graph::Cost_t> &SingleNodeVec,
                            NodesToReduceR1 &Nodes) {
  auto NumOfSelections = SingleNodeVec.h();
  auto MinCost = SingleNodeVec[0][0] + 
    Nodes.getCostForSelections(CostMatrix, 0u /*SingleNodeSelection*/, NeighbSelection);
  auto BestSelection = 0u;
  for (unsigned CurDependSelection = 0u; CurDependSelection < NumOfSelections;
       ++CurDependSelection) {
    auto CurCost = SingleNodeVec[CurDependSelection][0] + 
      Nodes.getCostForSelections(CostMatrix, CurDependSelection, NeighbSelection);
    if (CurCost < MinCost) {
      MinCost = CurCost;
      BestSelection = CurDependSelection;
    }
  }
  return {BestSelection, MinCost};
}

// R1 reduction:
//   Single -> Neighb -> ...
//               |
//               V
//              ...
//  Transforms to:
//            NewNeighb -> ...
//               |
//               V
//              ...
__device__ void __performR1Reduction(device::Graph &Graph,
                                     NodesToReduceR1 Nodes,
                                     unsigned *DependentSolutions,
                                     unsigned ThreadIdInReduction,
                                     unsigned ThreadsPerReduction) {
  auto Single = Nodes.getSingleNode();
  auto Neighb = Nodes.getNeighbour();
  auto [PosY, PosX] = Nodes.getPosOfCostMatix();
  auto &AdjMatrix = Graph.getAdjMatrix();
  auto CostMatrixIdx = AdjMatrix[PosY][PosX];
  auto &CostMatrix = Graph.getCostMatrix(CostMatrixIdx);
  auto &SingleNodeVec = Graph.getCostMatrix(Single);
  auto &NeighbNodeVec = Graph.getCostMatrix(Neighb);
  auto NumOfDefiningSolutions = NeighbNodeVec.h();

  assert(ThreadsPerReduction > 0);
  for (auto CurNeighbSelection = ThreadIdInReduction; 
       CurNeighbSelection < NumOfDefiningSolutions; 
       CurNeighbSelection += ThreadsPerReduction) {
    auto [BestSelection, AdditionalCost] = __getBestDependentSelection(CurNeighbSelection,
                                                     CostMatrix,
                                                     SingleNodeVec,
                                                     Nodes);
    DependentSolutions[CurNeighbSelection] = BestSelection;
    NeighbNodeVec[CurNeighbSelection][0] += AdditionalCost;
  }

  // Removes cost matrix and cost vector of the removed node
  if (ThreadIdInReduction == 0) {
    constexpr auto NoNode = -1;
    AdjMatrix[PosY][PosX] = NoNode;
    AdjMatrix[Single][Single] = NoNode;
  }
}

// Each R1 reduction performs with ThreadsPerReduction threads
__global__ void __R1Reduction(device::Graph Graph,
                              NodesToReduceR1 *NodesToReduce,
                              unsigned **DependentSolutions,
                              unsigned NumberOfReductions,
                              unsigned ThreadsPerReduction) {
  auto GlobalId = blockIdx.x * blockDim.x + threadIdx.x;
  auto ReductionId = GlobalId / ThreadsPerReduction;
  auto ThreadIdInReduction = GlobalId % ThreadsPerReduction;

  if (ReductionId < NumberOfReductions)
    __performR1Reduction(Graph, NodesToReduce[ReductionId],
                         DependentSolutions[ReductionId], ThreadIdInReduction,
                         ThreadsPerReduction);
}

__global__ void __getNodesToReduceR1(device::Graph Graph,
                                     NodesToReduceR1 *NodesToReduce) {
  auto &AdjMatrix = Graph.getAdjMatrix();
  auto GraphSize = AdjMatrix.h();
  auto GlobalId = blockIdx.x * blockDim.x + threadIdx.x;
  auto NoEdge = -1;

  if (GlobalId >= GraphSize)
    return;

  auto CurNodesToReduce = NodesToReduceR1{};
  auto NumOfNeighbours = 0u;
  for (unsigned i = 0; i < GraphSize; ++i) {
    if (i == GlobalId)
      continue;
    bool HasOutEdge = AdjMatrix[GlobalId][i] != NoEdge;
    bool HasInEdge = AdjMatrix[i][GlobalId] != NoEdge;
    NumOfNeighbours += HasInEdge;
    NumOfNeighbours += HasOutEdge;
    if (HasOutEdge) {
      CurNodesToReduce.addNeighbour(i);
      CurNodesToReduce.addAdjMatrixPos(GlobalId, i);
    }
    if (HasInEdge) {
      CurNodesToReduce.addNeighbour(i);
      CurNodesToReduce.addAdjMatrixPos(i, GlobalId);
    }
  }

  if (NumOfNeighbours != 1) {
    NodesToReduce[GlobalId] = NodesToReduceR1{};
    return;
  }
  CurNodesToReduce.addSingleNode(GlobalId);
  NodesToReduce[GlobalId] = CurNodesToReduce;
}

class DependentSolutionsForDevice final {
  struct DeviceDependentSolution final {
    // This is an selections to be taken on Dependent node
    //  based on selections of Defining node.
    thrust::device_vector<unsigned> DependentSelections;
    size_t DefiningHostIdx;
    size_t DependentHostIdx;
  };

  // defining selection to dependent one
  std::vector<DeviceDependentSolution> Solutions;
  thrust::device_vector<unsigned *> DeviceArrayOfReductions;

public:
  DependentSolutionsForDevice(
      device::Graph &GraphDevice, const Graph &Graph,
      thrust::host_vector<NodesToReduceR1> AllNodesToReduce) {
    DEBUG_EXPR(std::cout << "Current graph:\n");
    DEBUG_EXPR(GraphDevice.printAdjMatrix(std::cout));
    DEBUG_EXPR(std::cout << "Nodes to reduce R1:\n");
    for (auto &NodesToReduce : AllNodesToReduce) {
      auto DefiningNodeDeviceIdx = NodesToReduce.getNeighbour();
      auto DefiningNodeHostIdx = GraphDevice.getHostNode(DefiningNodeDeviceIdx);
      auto DependentHostIdx =
          GraphDevice.getHostNode(NodesToReduce.getSingleNode());
      DEBUG_EXPR(std::cout << NodesToReduce.getSingleNode() << " -- " 
                  << NodesToReduce.getNeighbour() << " -- ...\n");
      if (!DefiningNodeHostIdx || !DependentHostIdx)
        utils::reportFatalError("Node has already been reduced");
      auto DefiningSolutionsSize = Graph.getNodesCostSize(*DefiningNodeHostIdx);
      auto NewSolution = DeviceDependentSolution{
          thrust::device_vector<unsigned>(DefiningSolutionsSize),
          *DefiningNodeHostIdx, *DependentHostIdx};
      Solutions.emplace_back(std::move(NewSolution));
    }
    auto PointersToSolutions = thrust::host_vector<unsigned *>{};
    for (auto &DeviceSelectionVector : Solutions)
      PointersToSolutions.push_back(thrust::raw_pointer_cast(
          DeviceSelectionVector.DependentSelections.data()));
    DeviceArrayOfReductions = std::move(PointersToSolutions);
  }

  auto getPointerToSolutions() {
    return thrust::raw_pointer_cast(DeviceArrayOfReductions.data());
  }

  void addBounedSelections(Solution &Sol) {
    for (auto &DependentSolution : Solutions) {
      auto DefiningSelectionsToDependent =
          thrust::host_vector<unsigned>{DependentSolution.DependentSelections};
      Sol.addBoundedSolution(DependentSolution.DependentHostIdx,
                             DependentSolution.DefiningHostIdx,
                             {DefiningSelectionsToDependent.begin(),
                              DefiningSelectionsToDependent.end()});
    }
  }
};

thrust::host_vector<NodesToReduceR1> 
filterDependentNodes(thrust::host_vector<NodesToReduceR1> NodesToReduce) {
  auto Res = thrust::host_vector<NodesToReduceR1>{};
  auto UsedNodes = std::unordered_set<size_t>{};
  for (auto &Nodes : NodesToReduce) {
    if (!Nodes.canBeReduced())
      continue;
    auto SingleNode = Nodes.getSingleNode();
    auto NeighbNode = Nodes.getNeighbour();
    if (UsedNodes.count(SingleNode) || UsedNodes.count(NeighbNode))
      continue;
    UsedNodes.emplace(SingleNode);
    UsedNodes.emplace(NeighbNode);
    Res.push_back(Nodes);
  }
  return Res;
}

thrust::device_vector<NodesToReduceR1>
getNodesToReduceR1(device::Graph &CurGraph, unsigned BlockSize) {
  auto NumOfNodes = CurGraph.size();
  auto NodesToReduceDevice = thrust::device_vector<NodesToReduceR1>(NumOfNodes);
  dim3 ThrBlockDim{BlockSize};
  dim3 BlockGridDim{utils::ceilDiv(NumOfNodes, ThrBlockDim.x)};
  __getNodesToReduceR1<<<BlockGridDim, ThrBlockDim>>>(
      CurGraph, thrust::raw_pointer_cast(NodesToReduceDevice.data()));

  auto IndependentNodesToReduce = 
    filterDependentNodes(NodesToReduceDevice);
  return {IndependentNodesToReduce};
}

void commitR1ReductionToHost(thrust::host_vector<NodesToReduceR1> ReducedNodes,
                             device::Graph &GraphDevice,
                             DependentSolutionsForDevice &DependentSolutionsContainer, 
                             Solution &Sol) {
  // State of the device matrix should be changed in the device code
  //  so this funcntion should make host state the same as device one.
  DependentSolutionsContainer.addBounedSelections(Sol);
  for (auto &Reduced : ReducedNodes) {
    auto NodeToRemove = Reduced.getSingleNode();
    auto Neighb = Reduced.getNeighbour();
    GraphDevice.makeCostUnreachable(Neighb, NodeToRemove);
    GraphDevice.makeCostUnreachable(NodeToRemove, Neighb);
    GraphDevice.makeCostUnreachable(NodeToRemove, NodeToRemove);
  }
}

bool performR1Reduction(device::Graph &GraphDevice, const Graph &Graph,
                        Solution &Sol, unsigned BlockSize,
                        unsigned ThreadsPerReduction) {
  DEBUG_EXPR(std::cout << "\n---------------------\n");
  DEBUG_EXPR(std::cout << "Performing R1 reduction\n");
  auto OnlyNodesToReduce = getNodesToReduceR1(GraphDevice, BlockSize);
  DEBUG_EXPR(std::cout << "Found " << OnlyNodesToReduce.size()
                       << " nodes\n");
  if (OnlyNodesToReduce.size() == 0)
    return false;

  auto HostOnlyNodesToReduce =
      thrust::host_vector<NodesToReduceR1>(OnlyNodesToReduce);
  auto DependentSolutionsContainer = DependentSolutionsForDevice(
      GraphDevice, Graph, std::move(HostOnlyNodesToReduce));
  auto NumOfThreads = OnlyNodesToReduce.size() * ThreadsPerReduction;
  dim3 ThrBlockDim{BlockSize};
  dim3 BlockGridDim{utils::ceilDiv(NumOfThreads, ThrBlockDim.x)};
  __R1Reduction<<<BlockGridDim, ThrBlockDim>>>(
      GraphDevice, thrust::raw_pointer_cast(OnlyNodesToReduce.data()),
      DependentSolutionsContainer.getPointerToSolutions(),
      OnlyNodesToReduce.size(), ThreadsPerReduction);

  commitR1ReductionToHost(OnlyNodesToReduce, GraphDevice, 
                          DependentSolutionsContainer, Sol);

  return true;
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

GPUSolver::Res_t GPUSolver::PassManager::runPass(Pass_t &Pass, Res_t PrevRes,
                                                 Graph &Graph) {
  auto Start = std::chrono::steady_clock::now();
  auto Res = Pass->run(Graph, std::move(PrevRes));
  auto End = std::chrono::steady_clock::now();
  PassPtrToDuration[Pass.get()] += utils::to_milliseconds(End - Start);
  return Res;
}

size_t GPUSolver::PassManager::getNextIdx(LoopHeader &Header, Res_t &Res,
                                          size_t CurIdx) {
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
      CurStageIdx =
          getNextIdx(std::get<LoopHeader>(CurStage), Res, CurStageIdx);
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
  std::transform(Stages.begin(), Stages.end(), std::back_inserter(Res),
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
template <typename T> struct LoopState : public LoopCondition {
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

struct FinalMock final : public GPUSolver::FinalPass {
  Solution getSolution(const Graph &Graph, Res_t PrevResult) override {
    auto *ResPtr = dynamic_cast<LoopState<GPUGraphData> *>(PrevResult.get());
    if (!ResPtr)
      utils::reportFatalError("Invalid loop state in R0 reduction");
    auto &Sol = ResPtr->getMetadata().getSolution();
    DEBUG_EXPR(std::cout << "Solution:\n");
    DEBUG_EXPR(Sol.printSummary(std::cout));
    return Solution{};
  }
};

struct InitStatePass final : public GPUSolver::Pass {
  Res_t run(const Graph &Graph, Res_t PrevResult) override {
    auto Metadata =
        LoopState<GPUGraphData>::Metadata_t{new GPUGraphData(Graph)};
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
    DEBUG_EXPR(std::cout << "\n---------------------\n");
    DEBUG_EXPR(std::cout << "Graph has changed: " << Metadata.isGraphChanged() << "\n");
    ResPtr->setCondition(Metadata.isGraphChanged());
    Metadata.graphHasBeenChanged(false);
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
  bool Changed = performR0Reduction(Metadata.getDeviceGraph(), Graph,
                                    Metadata.getSolution(), BlockSize) ||
                 Metadata.isGraphChanged();
  Metadata.graphHasBeenChanged(Changed);
  return PrevResult;
}

GPUSolver::Res_t
HeuristicSolver::R1Reduction::run(const Graph &Graph,
                                  GPUSolver::Res_t PrevResult) {
  auto *ResPtr = dynamic_cast<LoopState<GPUGraphData> *>(PrevResult.get());
  if (!ResPtr)
    utils::reportFatalError("Invalid loop state in R0 reduction");
  auto &Metadata = ResPtr->getMetadata();
  bool Changed = performR1Reduction(Metadata.getDeviceGraph(), Graph,
                                    Metadata.getSolution(), BlockSize,
                                    ThreadsPerReduction) ||
                 Metadata.isGraphChanged();
  Metadata.graphHasBeenChanged(Changed);
  return PrevResult;
}

GPUSolver::Res_t
HeuristicSolver::CleanUpPass::run(const Graph &Graph,
                                  GPUSolver::Res_t PrevResult) {
  auto *ResPtr = dynamic_cast<LoopState<GPUGraphData> *>(PrevResult.get());
  if (!ResPtr)
    utils::reportFatalError("Invalid loop state in clean up");
  auto &Metadata = ResPtr->getMetadata();
  Metadata.getDeviceGraph().removeUnreachableNodes();
  return PrevResult;
}

void HeuristicSolver::addPasses(PassManager &PM) {
  PM.addPass(Pass_t{new InitStatePass});
  PM.addLoopStart(Condition_t{new LoopConditionHandler});
  PM.addPass(Pass_t{new R0Reduction});
  PM.addPass(Pass_t{new R1Reduction});
  PM.addPass(Pass_t{new CleanUpPass});
  PM.addPass(Pass_t{new GraphChangeChecker});
  PM.addLoopEnd();
  PM.addPass(Pass_t{new FinalMock});
}

} // namespace PBQP