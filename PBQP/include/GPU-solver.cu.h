#pragma once

#include "PBQP.h"
#include "GPU-graph.h"

namespace PBQP {

struct GPUSolver final : public Solver {
  struct Pass {
    
  };

  Solution solve(Graph Task) override {}
};

} // namespace PBQP