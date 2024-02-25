#include "PBQP.h"

#include <tuple>

namespace PBQP {

Graph::Edge::Edge(Node *Lhs, host::Matrix<Graph::Cost_t> CostMatrix, Node *Rhs) :
    Lhs{Lhs}, 
    Rhs{Rhs}, 
    CostMatrix{std::move(CostMatrix)} {
  assert(Lhs && Rhs);
  assert(Lhs->costSize() == CostMatrix.h());
  assert(Rhs->costSize() == CostMatrix.w());
}

Graph::Cost_t Graph::getCost(size_t LhsChoice, size_t RhsChoise) const {
  assert(LhsChoice < Lhs->costSize());
  assert(RhsChoice < Rhs->costSize());
  return CostMatrix[LhsChoice][RhsChoice];
}

std::pair<Graph::Node *, Graph::Node *> Graph::Edge::getNodes() const {
  return {Lhs, Rhs};
}

bool Graph::Edge::operator==(const Edge &RhsEdge) const {
  return std::tie(Lhs, Rhs) == std::tie(RhsEdge.Lhs, RhsEdge.Rhs);
}

std::unique_ptr<Graph::Edge> 
Graph::Node::createEdge(Graph::Node &Lhs, host::Matrix<Graph::Cost_t> CostMatrix,
                        Graph::Node &Rhs) {
  assert(Lhs.size() == CostMatrix.h());
  assert(Rhs.size() == CostMatrix.w());
  auto NewEdge = std::make_unique<Edge>(&Lhs, std::move(CostMatrix), &Rhs);
  Lhs.Edges.push_back(NewEdge.get());
  Rhs.Edges.push_back(NewEdge.get());
  return NewEdge;
}

void Graph::Edge::print(std::ostream &OS) const {
  for (size_t i = 0; i < CostMatrix.h(); ++i) {
    for (size_t j = 0; j < CostMatrix.w(); ++j)
      OS << CostMatrix[i][j] << " ";
    OS << "\n";
  }
}

void Graph::Node::print(std::ostream &OS) const {
  for (auto Cost : CostVector)
    OS << Cost << "\n";
}

void Graph::print(std::ostream &OS) const {
  OS << "digraph Dump {\n" <<
        "node[" <<  GraphNodeColour << "]\n";
  for (auto &Node : Nodes) {
    OS << "\"" << &Node << "\" label = \"";
    Node.print(OS);
    OS << "\"]\n";
  }

  for (auto &Edge : Edges) {
    auto [Lhs, Rhs] = Edge.getNodes();
    OS << "\"" << &Lhs << "\" -- \"" << &Rhs << " [label = \"";
    Edge.print(OS);
    OS << "\"]\n"; 
  }
  OS << "}\n";
}
} // namespace PBQP 