#include "GPU-solver.cu.h"

namespace PBQP {

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
  PM.addPass(Pass_t{new Mock});

  return PM.run(std::move(Task));           
}

namespace {

// Class for passing device::Graph through PassManager
struct GPUGraph final : public GPUSolver::Pass::Result {
  device::Graph Graph;

  GPUGraph(const PBQP::Graph &HostGraph) : Graph{HostGraph} {}
};

// Pass which creates device::Graph and passes it further
class GraphLoader final : public GPUSolver::Pass {
  GPUGraph *loadGraphToGPU(const Graph &Graph) {
    return new GPUGraph(Graph);
  }

public:
  Res_t run(const Graph &Graph, Res_t PrevResult) override {
    return Res_t{loadGraphToGPU(Graph)};
  }
};


class FullSearchImpl final : public FinalPass {
  
  Solution getSolution(const Graph &Graph, Res_t PrevResult) override {
    auto *GPUGraph = dynamic_cast<GPUGraph *>(PrevResult.get());
    if (!GPUGraph)
      utils::reportFatalError("Graph hasn't been loaded to GPU");
    
  }
};

} // anonymous namespace

Solution GPUFullSearch::getBestOption(const Graph &Graph) {
  
}

} // namespace PBQP