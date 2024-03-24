#include "Graph-gen.h"

#include <fstream>

int main() {
  auto Graph = PBQP::generateGraph(PBQP::GenConfig{});
  auto DotGraphOS = std::ofstream{"graph-gen.dot"};
  Graph.print(DotGraphOS);
  assert(Graph.validate());
}