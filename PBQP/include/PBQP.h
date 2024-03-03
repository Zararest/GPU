#pragma once

#include "Matrix.h"

#include <limits>
#include <memory>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <map>

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
    std::pair<size_t, size_t> dimension() const;

    bool operator==(const Edge &RhsEdge) const;

    void print(std::ostream &OS) const;
  };

  class Node {
    host::Matrix<Cost_t> CostVector;
    std::vector<Edge*> Edges;

  public:
    Node(host::Matrix<Cost_t> CostVector) : CostVector{std::move(CostVector)} {}

    static std::unique_ptr<Edge> createEdge(Node &Lhs, host::Matrix<Cost_t> CostMatrix, Node &Rhs);

    void changeCost(host::Matrix<Cost_t> NewCostVector);
    size_t costSize() const { return CostVector.h(); }
    size_t order() const { return Edges.size(); }

    void print(std::ostream &OS) const;
  };

private:
  std::vector<std::unique_ptr<Node>> Nodes;
  std::vector<std::unique_ptr<Edge>> Edges;
  static constexpr std::string_view GraphNodeColour = 
    "color=olive,fontsize=14, style=filled, shape=rectangle";

  void parseNode(std::istream &IS, std::map<size_t, size_t> &AddrToNodexIdx);
  void parseEdge(std::istream &IS, std::map<size_t, size_t> &AddrToNodexIdx);
  Node &getNodeByAddr(size_t Addr, std::map<size_t, size_t> &AddrToNodexIdx);

public:
  Node &addNode(host::Matrix<Cost_t> CostVector) {
    Nodes.emplace_back(std::make_unique<Node>(std::move(CostVector)));
    return *Nodes.back();
  }

  Edge &addEdge(Node &Lhs, host::Matrix<Cost_t> CostMatrix, Node &Rhs) {
    Edges.emplace_back(Node::createEdge(Lhs, std::move(CostMatrix), Rhs));
    return *Edges.back();
  }

  //print graphviz
  void print(std::ostream &OS) const;

  //dump graph in internal representation
  void dump(std::ostream &OS) const;

  void read(std::istream &IS);
};

class Solution final {

};

class Solver {

public:
  virtual Solution solve(Graph Task) = 0;
};

} // namespace PBQP 