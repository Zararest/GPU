#pragma once

#include "PBQP.h"

namespace PBQP {

struct CPUFullSearch final : public Solver {
  
  Solution solve(Graph Task) override;
};

} // namespace PBQP 