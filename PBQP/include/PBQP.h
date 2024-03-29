#pragma once

#include "Matrix.h"

#include <limits>
#include <memory>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <map>
#include <optional>

namespace PBQP {

struct Graph final {
  using Cost_t = float;
  static constexpr Cost_t InfCost = std::numeric_limits<Cost_t>::infinity();
  static constexpr auto InfLiteral = "inf";
  class Node;

  // Lhs{nL} CostMatrix{nL x nR} Rhs{nR} 
  class Edge final {
    host::Matrix<Cost_t> CostMatrix;
    Node *Lhs = nullptr;
    Node *Rhs = nullptr;

  public:
    Edge(Node *Lhs, host::Matrix<Cost_t> CostMatrix, Node *Rhs);
    Cost_t getCost(size_t LhsChoice, size_t RhsChoise) const;
    const host::Matrix<Cost_t> &getCostMatrix() const { return CostMatrix; }
    std::pair<Node *, Node *> getNodes() const;
    std::pair<size_t, size_t> dimension() const;

    bool operator==(const Edge &RhsEdge) const;

    void print(std::ostream &OS) const;
  };

  class Node final {
    host::Matrix<Cost_t> CostVector;
    std::vector<Edge*> Edges;
    std::string Name;

  public:
    Node(host::Matrix<Cost_t> CostVector, std::string Name = "node") : 
      CostVector{std::move(CostVector)}, Name{Name} {}

    static std::unique_ptr<Edge> createEdge(Node &Lhs, 
                                            host::Matrix<Cost_t> CostMatrix, 
                                            Node &Rhs);

    void changeCost(host::Matrix<Cost_t> NewCostVector);
    void changeName(std::string NewName) { Name = std::move(NewName); }
    const host::Matrix<Cost_t> &getCostVector() const { return CostVector; }
    Cost_t getCost(size_t Choice) const { return CostVector[Choice][0]; }
    const std::string &getName() const { return Name; }
    size_t costSize() const { return CostVector.h(); }
    size_t order() const { return Edges.size(); }
    auto edgesBeg() { return Edges.begin(); }
    auto edgesEnd() { return Edges.end(); }
    auto edgesBeg() const { return Edges.begin(); }
    auto edgesEnd() const { return Edges.end(); }

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
  
  //cmp by ptr
  static bool nodeHasEdge(const Node &Node, const Edge &Edge);
  static bool edgeHasNode(const Edge &Edge, const Node &Node);

public:
  // For better error handling
  Graph() = default;
  Graph(const Graph &) = delete;
  Graph(Graph&&) = default;
  ~Graph() = default;

  Graph &operator=(const Graph&) = delete;
  Graph &operator=(Graph&&) = default;

  Node &addNode(host::Matrix<Cost_t> CostVector) {
    Nodes.emplace_back(std::make_unique<Node>(std::move(CostVector)));
    return *Nodes.back();
  }

  Edge &addEdge(Node &Lhs, host::Matrix<Cost_t> CostMatrix, Node &Rhs) {
    Edges.emplace_back(Node::createEdge(Lhs, std::move(CostMatrix), Rhs));
    return *Edges.back();
  }

  static Graph copy(const Graph &OldGraph);
  static Graph merge(const Graph &LhsClique, const Graph &RhsClique);

  size_t size() const { return Nodes.size(); }
  auto nodesBeg() const { return Nodes.begin(); }
  auto nodesEnd() const { return Nodes.end(); }
  auto edgesBeg() const { return Edges.begin(); }
  auto edgesEnd() const { return Edges.end(); }

  bool validate() const;
  //print graphviz
  void print(std::ostream &OS) const;
  //dump graph in internal representation
  void dump(std::ostream &OS) const;
  //construct graph from internal representation
  void read(std::istream &IS);
};

class Solution final {
  // intermediate solution might have no graph in it
  std::optional<Graph> InitialGraph;
  //node's index to choise
  std::map<size_t, size_t> SelectedVariants;
  Graph::Cost_t FinalCost = 0;
  static constexpr std::string_view AnswerNodeColour =
    "color=coral, fontsize=18, style=filled, shape=oval";
  static constexpr std::string_view SolutionColour = 
    "color=red, fontsize=14, style=filled, shape=oval";

public:
  Solution() = default;
  Solution(Graph InitialGraph) : InitialGraph{std::move(InitialGraph)} {}

  bool isFinal() const { return InitialGraph.has_value();}
  void makeFinal(Graph InitialGraphIn);
  const Graph &getGraph() const;
  void clear() { SelectedVariants.clear(); }
  bool addSelection(size_t NodeIdx, size_t Select) {
    return SelectedVariants.insert({NodeIdx, Select}).second;
  }
  void addFinalCost(Graph::Cost_t NewFinalCost) {
    FinalCost = NewFinalCost;
  }
  Graph::Cost_t getFinalCost() const {
    return FinalCost;
  }
  void print(std::ostream &OS) const;
};

struct Solver {
  virtual Solution solve(Graph Task) = 0;
  virtual ~Solver() {}
};

} // namespace PBQP 