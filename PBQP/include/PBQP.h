#pragma once

#include "Matrix.h"

#include <limits>
#include <memory>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

namespace PBQP {

struct Graph final {
  using Cost_t = float;
  class Node;

  // Lhs{nL} CostMatrix{nL x nR} Rhs{nR} 
  class Edge {
    host::Matrix<Cost_t> CostMatrix;
    Node *Lhs = nullptr;
    Node *Rhs = nullptr;

  public:
    static constexpr Cost_t InfCost = std::numeric_limits<Cost_t>::quiet_NaN();

    Edge(Node *Lhs, host::Matrix<Cost_t> CostMatrix, Node *Rhs);
    Cost_t getCost(size_t LhsChoice, size_t RhsChoise) const;
    std::pair<Node *, Node *> getNodes() const;

    bool operator==(const Edge &RhsEdge) const;

    void print(std::ostream &OS) const;
  };

  class Node {
    host::Matrix<Cost_t> CostVector;
    std::vector<Edge*> Edges;

  public:
    Node(host::Matrix<Cost_t> CostVector) : CostVector{std::move(CostVector)} {}

    static std::unique_ptr<Edge> createEdge(Node &Lhs, host::Matrix<Cost_t> CostMatrix, Node &Rhs);

    size_t costSize() const { return CostVector.size(); }
    size_t order() const { return Edges.size(); }

    void print(std::ostream &OS) const;
  };

private:
  std::vector<std::unique_ptr<Node>> Nodes;
  std::vector<std::unique_ptr<Edge>> Edges;
  static constexpr std::string_view GraphNodeColour = 
    "color=red,fontsize=14, style=filled";

public:
  Node &addNode(host::Matrix<Cost_t> CostVector) {
    Nodes.emplace_back(std::make_unique<Node>(std::move(CostVector)));
    return *Nodes.back();
  }

  Edge &addEdge(Node &Lhs, host::Matrix<Cost_t> CostMatrix, Node &Rhs) {
    Edges.emplace_back(std::make_unique<Edge>(&Lhs, std::move(CostMatrix), &Rhs));
    return *Edges.back();
  }

  void print(std::ostream &OS) const;
};

class Solution final {

};

class Solver {

public:
  virtual Solution solve(Graph Task) = 0;
};

} // namespace PBQP 