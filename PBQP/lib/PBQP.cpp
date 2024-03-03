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

Graph::Cost_t Graph::Edge::getCost(size_t LhsChoice, size_t RhsChoice) const {
  assert(LhsChoice < Lhs->costSize());
  assert(RhsChoice < Rhs->costSize());
  return CostMatrix[LhsChoice][RhsChoice];
}

std::pair<Graph::Node *, Graph::Node *> Graph::Edge::getNodes() const {
  return {Lhs, Rhs};
}

std::pair<size_t, size_t> Graph::Edge::dimension() const {
  return {CostMatrix.h(), CostMatrix.w()};
}

bool Graph::Edge::operator==(const Edge &RhsEdge) const {
  return std::tie(Lhs, Rhs) == std::tie(RhsEdge.Lhs, RhsEdge.Rhs);
}

std::unique_ptr<Graph::Edge> 
Graph::Node::createEdge(Graph::Node &Lhs, host::Matrix<Graph::Cost_t> CostMatrix,
                        Graph::Node &Rhs) {
  assert(Lhs.costSize() == CostMatrix.h());
  assert(Rhs.costSize() == CostMatrix.w());
  auto NewEdge = std::make_unique<Edge>(&Lhs, std::move(CostMatrix), &Rhs);
  Lhs.Edges.push_back(NewEdge.get());
  Rhs.Edges.push_back(NewEdge.get());
  return NewEdge;
}

void Graph::Node::changeCost(host::Matrix<Graph::Cost_t> NewCostVector) {
  CostVector = std::move(NewCostVector);
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
  OS << "graph Dump {\n" <<
        "node[" <<  GraphNodeColour << "]\n";
  for (auto &Node : Nodes) {
    assert(Node);
    OS << "\"" << Node.get() << "\" [label = \"";
    Node->print(OS);
    OS << "\"]\n";
  }

  for (auto &Edge : Edges) {
    assert(Edge);
    auto [Lhs, Rhs] = Edge->getNodes();
    OS << "\"" << Lhs << "\" -- \"" << Rhs << "\" [label = \"";
    Edge->print(OS);
    OS << "\"]\n"; 
  }
  OS << "}\n";
}

void Graph::dump(std::ostream &OS) const {
  OS << "Edges: " << Edges.size() << "\n";
  for (auto &Edge : Edges) {
    assert(Edge);
    auto [Lhs, Rhs] = Edge->getNodes();
    auto [H, W] = Edge->dimension();
    OS << H << " x " << W << "\n";
    Edge->print(OS);
    OS << Lhs << " " << Rhs << "\n";
  }
  OS << "\n";

  OS << "Nodes: " << Nodes.size() << "\n";
  for (auto &Node : Nodes) {
    assert(Node);
    OS << Node.get() << " " << Node->costSize() << " x 1\n";
    Node->print(OS);
    OS << "\n";
  }
}

static size_t getSize(std::istream &IS) {
  auto Val = 0ull;
  if (!(IS >> Val))
    utils::reportFatalError("Can't read a matrix size");
  return Val;
}

static host::Matrix<Graph::Cost_t> parseMatrix(std::istream &IS) {
  auto Separator = std::string{};
  auto Hight = getSize(IS);
  
  IS >> Separator;
  if (Separator != "x")
    utils::reportFatalError("Wrong matrix size separator");

  auto Width = getSize(IS);
  auto Data = std::vector<Graph::Cost_t>{};
  for (size_t i = 0; i < Hight * Width; ++i) {
    auto Val = Graph::Cost_t{};
    if (!(IS >> Val))
      utils::reportFatalError("Can't read a matrix val");
    Data.push_back(Val);
  }
  return {Data.begin(), Data.end(), Hight, Width};
}

Graph::Node &Graph::getNodeByAddr(size_t Addr, std::map<size_t, size_t> &AddrToNodexIdx) {
  auto NodeIt = AddrToNodexIdx.find(Addr);
  if (NodeIt != AddrToNodexIdx.end()) {
    assert(NodeIt->second < Nodes.size());
    return *Nodes[NodeIt->second];
  }
  AddrToNodexIdx[Addr] = Nodes.size();
  return addNode(host::Matrix<Cost_t>{});
}

static size_t getAddr(std::istream &IS) {
  auto Addr = 0ull;
  if (!(IS >> std::hex >> Addr))
    utils::reportFatalError("Can't read a node's address");
  return Addr;
}

void Graph::parseEdge(std::istream &IS, std::map<size_t, size_t> &AddrToNodexIdx) {
  auto CostMatrix = parseMatrix(IS);
  auto LhsAddr = getAddr(IS);
  auto RhsAddr = getAddr(IS);
  auto &LhsNode = getNodeByAddr(LhsAddr, AddrToNodexIdx);
  auto &RhsNode = getNodeByAddr(RhsAddr, AddrToNodexIdx);
  LhsNode.changeCost(host::Matrix<Cost_t>{CostMatrix.h()});
  RhsNode.changeCost(host::Matrix<Cost_t>{CostMatrix.w()});
  Edges.emplace_back(Node::createEdge(LhsNode, std::move(CostMatrix), RhsNode));
}

static size_t parseNumberOfItems(std::istream &IS, const std::string &Name) {
  auto Word = std::string{};
  IS >> Word;
  if (Word != Name + ":")
      utils::reportFatalError("Can't parse number of " + Name);
  
  auto NumOfItems = 0ul;
  if (!(IS >> NumOfItems))
    utils::reportFatalError("Can't parse number of " + Name);
  return NumOfItems;
}

void Graph::parseNode(std::istream &IS, std::map<size_t, size_t> &AddrToNodexIdx) {
  auto NodeAddr = getAddr(IS);
  auto CostVector = parseMatrix(IS);
  auto &Node = getNodeByAddr(NodeAddr, AddrToNodexIdx);
  Node.changeCost(std::move(CostVector));
}

void Graph::read(std::istream &IS) { 
  auto AddrToNodeIdx = std::map<size_t, size_t>{};
  auto NumOfEdges = parseNumberOfItems(IS, "Edges");
  for (size_t i = 0; i < NumOfEdges; ++i)
    parseEdge(IS, AddrToNodeIdx);

  auto NumOfNodes = parseNumberOfItems(IS, "Nodes");
  for (size_t i = 0; i < NumOfNodes; ++i)
    parseNode(IS, AddrToNodeIdx);
}
} // namespace PBQP 