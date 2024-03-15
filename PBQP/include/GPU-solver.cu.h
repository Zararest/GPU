#pragma once

#include "PBQP.h"
#include "GPU-graph.h"

namespace PBQP {

// Base for every GPU solver
struct GPUSolver : public Solver {
  // Class for graph transformation
  struct Pass {
    struct Result {
      virtual ~Result() = 0;
    };

    using Res_t = std::unique_ptr<Result>;

    virtual Res_t run(const Graph &Graph, Res_t PrevResult) = 0;
    virtual ~Pass() {}
  };

  using Pass_t = std::unique_ptr<Pass>;
  
  // Class to run all transformations
  class PassManager final {
    std::vector<Pass_t> Passes;

  public:
    void addPass(Pass_t Pass);
    Solution run(Graph Graph);
  };

  // Wrapper for PBQP::Solution.
  // Last pass in pipeline should return this object as a result.
  class FinalSolution final : public Pass::Result {
    Solution Sol;

  public:
    FinalSolution(Solution NewSolution) : Sol{std::move(NewSolution)} {}

    Solution getFinalSolution(Graph Graph) {
      Sol.makeFinal(std::move(Graph));
      return std::move(Sol);
    }
  };

  // Last pass in a pipeline
  struct FinalPass : public Pass { 

    virtual Solution getSolution(const Graph &Graph, Res_t PrevResult) = 0;

    Res_t run(const Graph &Graph, Res_t PrevResult) {
      auto Solution = this->getSolution(Graph, std::move(PrevResult));
      return Res_t{new FinalSolution(std::move(Solution))};
    }
    
    virtual ~FinalPass() {}
  };

  // Extension point for derivative classes
  virtual void addPasses(PassManager &PM) = 0;
  Solution solve(Graph Task) override;
  virtual ~GPUSolver() {}
};

// Mock solver
struct GPUMock final : public GPUSolver {
  struct FinalPass : public GPUSolver::FinalPass {
    Solution getSolution(const Graph &Graph, Res_t PrevResult) override {
      return Solution{};
    }
  };

  void addPasses(PassManager &PM) override {
    PM.addPass(Pass_t{new FinalPass});
  }
};

// GPU solver with full search of optimal solution
struct GPUFullSearch final : public GPUSolver {  
  void addPasses(PassManager &PM) override;
};

} // namespace PBQP