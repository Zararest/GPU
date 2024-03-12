#include "GPU-solver.cu.h"

namespace PBQP {

GPUSolver::Pass::Result::~Result() {}

void GPUSolver::PassManager::addPass(Pass_t Pass) {
  Passes.emplace_back(std::move(Pass));
}

Solution GPUSolver::PassManager::run(const Graph &Graph) {
  auto Res = Pass::Res_t{};
  for (auto &Pass : Passes) {
    Res = Pass->run(Graph, std::move(Res));
  }
  auto *SolutionPtr = dynamic_cast<FinalSolution *>(Res.get());
  if (!SolutionPtr)
    utils::reportFatalError("Invalid final pass");
  return SolutionPtr->getSolution();
}


Solution GPUSolver::solve(Graph Task) {
  auto PM = PassManager{};
  PM.addPass(Pass_t{new Mock});

  return PM.run(Task);           
}

} // namespace PBQP