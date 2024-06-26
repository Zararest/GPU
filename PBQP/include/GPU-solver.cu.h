#pragma once

#include "GPU-graph.h"
#include "PBQP.h"

#include <variant>

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

  // Class for loops in pass manager
  struct Condition {
    virtual bool check(Pass::Res_t &PrevResult) = 0;
    virtual ~Condition() {}
  };

  using Res_t = Pass::Res_t;
  using Pass_t = std::unique_ptr<Pass>;
  using Condition_t = std::unique_ptr<Condition>;

  // Class to run all transformations
  class PassManager final {
    // If Condition::check() returns true, then pass in loop is being executed,
    //   otherwise pass after corresponding LoopEnd is being executed
    struct LoopHeader final {
      Condition_t Cond;
      size_t EndIdx = 0;
    };

    struct LoopEnd final {
      size_t HeaderIdx = 0;
    };

    std::vector<std::variant<Pass_t, LoopHeader, LoopEnd>> Stages;
    std::map<Pass *, size_t> PassPtrToDuration;
    std::map<Pass *, std::string> PassPtrToName;
    std::map<const LoopHeader *, size_t> LoopPtrToIterNum;
    size_t NumOfIncompleteLoops = 0;
    std::vector<size_t> LoopHeaderIdxes;

    size_t getNextIdx(LoopHeader &Header, Res_t &Res, size_t CurIdx);
    Res_t runPass(Pass_t &Pass, Res_t PrevRes, Graph &Graph);
    void addHeaderMetadata(LoopHeader &Header);

  public:
    using Profile_t = std::vector<std::pair<std::string, size_t>>;

    void addPass(Pass_t Pass, std::string Name = "");
    void addLoopStart(Condition_t Cond);
    void addLoopEnd();
    Solution run(Graph Graph);
    Profile_t getProfileInfo() const;
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

protected:
  PassManager PM;

public:
  // Extension point for derivative classes
  virtual void addPasses(PassManager &PM) = 0;
  Solution solve(Graph Task) override;
  PassManager::Profile_t getProfileInfo() const { return PM.getProfileInfo(); }
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

class ReductionsSolver final : public GPUSolver {

  // Pass that performs R0 reduction on graph:
  //  finds nodes with degree 0 and selects best option
  struct R0Reduction final : public GPUSolver::Pass {
    static constexpr auto BlockSize = 32ul;

    Res_t run(const Graph &Graph, Res_t PrevResult) override;
  };

  // Pass that performs R0 reduction on graph:
  //  finds a node that has only one neighbor
  //  and moves all costs to the neighbor
  struct R1Reduction final : public GPUSolver::Pass {
    static constexpr auto BlockSize = 32ul;
    static constexpr auto ThreadsPerReduction = 8ul;

    Res_t run(const Graph &Graph, Res_t PrevResult) override;
  };

  struct RNReduction final : public GPUSolver::Pass {
    static constexpr auto BlockSize = 16ul;
    // Max number of combinations to be considered
    static constexpr auto MaxNumOfCombinations = 8192ul;

    Res_t run(const Graph &Graph, Res_t PrevResult) override;
  };

  struct FinalFullSearch final : public GPUSolver::Pass {
    static constexpr auto BlockSize = 32ul;
    static constexpr auto MaxNumOfCombinations = 8192ul;

    Res_t run(const Graph &Graph, Res_t PrevResult) override;
  };

  // This is a pass that removes nodes that has been reduced
  struct CleanUpPass final : public GPUSolver::Pass {
    Res_t run(const Graph &Graph, Res_t PrevResult) override;
  };

public:
  void addPasses(PassManager &PM) override;
};
} // namespace PBQP