#pragma once

#include "PBQP.h"
#include "GPU-graph.h"

namespace PBQP {

struct GPUSolver final : public Solver {
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
    Solution run(const Graph &Graph);
  };

  Solution solve(Graph Task) override;
};

class FinalSolution final : public GPUSolver::Pass::Result {
  Solution Sol;

public:
  FinalSolution(Solution NewSolution) : Sol{std::move(NewSolution)} {}

  Solution getSolution() {
    return std::move(Sol);
  }
};

struct FinalPass : public GPUSolver::Pass {

  virtual Solution getSolution(const Graph &Graph, Res_t PrevResult) = 0;

  Res_t run(const Graph &Graph, Res_t PrevResult) {
    auto Solution = this->getSolution(Graph, std::move(PrevResult));
    return Res_t{new FinalSolution(std::move(Solution))};
  }
  
  virtual ~FinalPass() {}
};

struct Mock final : public FinalPass {
  Solution getSolution(const Graph &Graph, Res_t PrevResult) override {
    return Solution{Graph::copy(Graph)};
  }
};

} // namespace PBQP