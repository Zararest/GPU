#pragma once

#include "PBQP.h"
#include "GPU-graph.h"

namespace PBQP {

struct GPUSolver : public Solver {
  struct Pass {
    struct Result {
      virtual ~Result() = 0;
    };

    using Res_t = std::unique_ptr<Result>;

    virtual Res_t run(const Graph &Graph, Res_t PrevResult) = 0;
    virtual ~Pass() {}
  };

  using Pass_t = std::unique_ptr<Pass>;

  class PassManager final {
    std::vector<Pass_t> Passes;

  public:
    void addPass(Pass_t Pass);
    Solution run(Graph Graph);
  };

  Solution solve(Graph Task) override;

  class FinalSolution final : public Result {
    Solution Sol;

  public:
    FinalSolution(Solution NewSolution) : Sol{std::move(NewSolution)} {}

    Solution getFinalSolution(Graph Graph) {
      Sol.makeFinal(std::move(Graph));
      return std::move(Sol);
    }
  };

  struct FinalPass : public Pass {

    virtual Solution getSolution(const Graph &Graph, Res_t PrevResult) = 0;

    Res_t run(const Graph &Graph, Res_t PrevResult) {
      auto Solution = this->getSolution(Graph, std::move(PrevResult));
      return Res_t{new FinalSolution(std::move(Solution))};
    }
    
    virtual ~FinalPass() {}
  };
};

struct Mock final : public GPUSolver::FinalPass {
  Solution getSolution(const Graph &Graph, Res_t PrevResult) override {
    return Solution{};
  }
};

class GPUFullSearch final : GPUSolver::FinalPass {
  Solution getBestOption(const Graph &Graph);

public:
  Solution getSolution(const Graph &Graph, Res_t PrevResult) override {
    return getBestOption(Graph);
  }
};

} // namespace PBQP