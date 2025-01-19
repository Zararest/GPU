#include "PBQP.h"

#include <sstream>
#include <tuple>
#include <unordered_map>
#include <regex>

namespace PBQP {

Graph::Edge::Edge(Node *Lhs, host::Matrix<Graph::Cost_t> CostMatrix, Node *Rhs)
    : Lhs{Lhs}, Rhs{Rhs}, CostMatrix{std::move(CostMatrix)} {
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
Graph::Node::createEdge(Graph::Node &Lhs,
                        host::Matrix<Graph::Cost_t> CostMatrix,
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
  utils::printMatrix(CostMatrix, OS);
}

void Graph::Node::print(std::ostream &OS) const {
  for (auto Cost : CostVector)
    OS << Cost << "\n";
  OS << Name << "\n";
}

Graph Graph::copy(const Graph &OldGraph) {
  auto NewGraph = Graph{};
  std::transform(OldGraph.Nodes.begin(), OldGraph.Nodes.end(),
                 std::back_inserter(NewGraph.Nodes), [](const auto &NodePtr) {
                   assert(NodePtr);
                   return std::make_unique<Node>(NodePtr->getCostVector(),
                                                 NodePtr->getName());
                 });
  auto NodeAddrToIdx = std::unordered_map<Node *, size_t>{};
  for (size_t Idx = 0; Idx < OldGraph.Nodes.size(); ++Idx)
    NodeAddrToIdx.insert({OldGraph.Nodes[Idx].get(), Idx});

  for (auto &Edge : OldGraph.Edges) {
    auto [LhsNodeAddr, RhsNodeAddr] = Edge->getNodes();
    assert(NodeAddrToIdx.find(LhsNodeAddr) != NodeAddrToIdx.end());
    assert(NodeAddrToIdx.find(RhsNodeAddr) != NodeAddrToIdx.end());
    auto &LhsNodePtr = NewGraph.Nodes[NodeAddrToIdx[LhsNodeAddr]];
    auto &RhsNodePtr = NewGraph.Nodes[NodeAddrToIdx[RhsNodeAddr]];
    assert(LhsNodePtr && RhsNodePtr);
    NewGraph.addEdge(*LhsNodePtr, Edge->getCostMatrix(), *RhsNodePtr);
  }
  return NewGraph;
}

Graph Graph::merge(const Graph &LhsClique, const Graph &RhsClique) {
  auto NewLhs = Graph::copy(LhsClique);
  auto NewRhs = Graph::copy(RhsClique);
  auto FinalGraph = Graph{};
  auto MoveNode = [](std::unique_ptr<Node> &Node) { return std::move(Node); };
  auto MoveEdge = [](std::unique_ptr<Edge> &Edge) { return std::move(Edge); };
  std::transform(NewLhs.Nodes.begin(), NewLhs.Nodes.end(),
                 std::back_inserter(FinalGraph.Nodes), MoveNode);
  std::transform(NewLhs.Edges.begin(), NewLhs.Edges.end(),
                 std::back_inserter(FinalGraph.Edges), MoveEdge);

  std::transform(NewRhs.Nodes.begin(), NewRhs.Nodes.end(),
                 std::back_inserter(FinalGraph.Nodes), MoveNode);
  std::transform(NewRhs.Edges.begin(), NewRhs.Edges.end(),
                 std::back_inserter(FinalGraph.Edges), MoveEdge);
  return FinalGraph;
}

bool Graph::nodeHasEdge(const Node &Node, const Edge &Edge) {
  return std::find(Node.edgesBeg(), Node.edgesEnd(), &Edge) != Node.edgesEnd();
}

bool Graph::edgeHasNode(const Edge &Edge, const Node &Node) {
  auto [Lhs, Rhs] = Edge.getNodes();
  return Lhs == &Node || Rhs == &Node;
}

bool Graph::validate() const {
  for (auto &Edge : Edges) {
    const auto [Lhs, Rhs] = Edge->getNodes();
    assert(Lhs && Rhs);
    assert(Edge);
    if (!nodeHasEdge(*Lhs, *Edge) || !nodeHasEdge(*Rhs, *Edge))
      return false;
  }

  for (auto &Node : Nodes)
    if (std::any_of(Node->edgesBeg(), Node->edgesEnd(),
                    [&Node](const Edge *EdgePtr) {
                      assert(EdgePtr);
                      assert(Node);
                      return !edgeHasNode(*EdgePtr, *Node);
                    }))
      return false;
  return true;
}

void Graph::print(std::ostream &OS) const {
  OS << "digraph Dump {\n"
     << "node[" << GraphNodeColour << "]\n";
  for (auto &Node : Nodes) {
    assert(Node);
    OS << "\"" << Node.get() << "\" [label = \"";
    Node->print(OS);
    OS << "\"]\n";
  }

  for (auto &Edge : Edges) {
    assert(Edge);
    auto [Lhs, Rhs] = Edge->getNodes();
    OS << "\"" << Lhs << "\" -> \"" << Rhs << "\" [label = \"";
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

static Graph::Cost_t getValue(std::istream &IS) {
  auto ValStr = std::string{};
  IS >> ValStr;
  if (ValStr == Graph::InfLiteral)
    return Graph::InfCost;
  auto SS = std::stringstream{ValStr};
  auto Val = Graph::Cost_t{};
  if (!(SS >> Val))
    utils::reportFatalError("Can't read a matrix's value [" + ValStr + "]");
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
  for (size_t i = 0; i < Hight * Width; ++i)
    Data.push_back(getValue(IS));
  return {Data.begin(), Data.end(), Hight, Width};
}

Graph::Node &Graph::getNodeByAddr(size_t Addr,
                                  std::map<size_t, size_t> &AddrToNodexIdx) {
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
  IS >> std::dec;
  return Addr;
}

void Graph::parseEdge(std::istream &IS,
                      std::map<size_t, size_t> &AddrToNodexIdx) {
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

static std::string parseName(std::istream &IS) {
  auto Name = std::string{};
  if (!(IS >> Name))
    utils::reportFatalError("Can't parse node's name");
  return Name;
}

void Graph::parseNode(std::istream &IS,
                      std::map<size_t, size_t> &AddrToNodexIdx) {
  auto NodeAddr = getAddr(IS);
  auto CostVector = parseMatrix(IS);
  auto Name = parseName(IS);
  auto &Node = getNodeByAddr(NodeAddr, AddrToNodexIdx);
  Node.changeCost(std::move(CostVector));
  Node.changeName(std::move(Name));
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

const Graph &Solution::getGraph() const {
  if (!InitialGraph)
    utils::reportFatalError("Initial graph has not been loaded yet");
  return *InitialGraph;
}

void Solution::resolveBoundedSolutions() {
  auto Changed = true;
  DEBUG_EXPR(std::cout << "Bounded solutions: \n");
  DEBUG_EXPR(for (auto &BoundedSol : BoundedSolutions) 
                BoundedSol.print(std::cout););
  while (Changed) {
    DEBUG_EXPR(std::cout << "In loop\n");
    auto NewSolutons = std::unordered_map<size_t, size_t>{};
    for (auto &BoundedSol : BoundedSolutions) {
      auto DefiningNode = BoundedSol.getDefiningNode();
      if (SelectedVariants.find(DefiningNode) == SelectedVariants.end())
        continue;
      auto [NewResolvedNode, NewNodeSol] =
          BoundedSol.getDependentSolution(SelectedVariants[DefiningNode]);
      NewSolutons.emplace(NewResolvedNode, NewNodeSol);
    }
    Changed = !NewSolutons.empty();
    for (auto [NewResolvedNode, NewNodeSol] : NewSolutons) {
      if (!addSelection(NewResolvedNode, NewNodeSol))
        utils::reportFatalError("Solution already has been added");
      BoundedSolutions.erase(BoundedSolution{NewResolvedNode, NewNodeSol});
    }
  }

  if (!BoundedSolutions.empty())
    utils::reportFatalError("There are unresolved noeds");
}

Graph::Cost_t Solution::calcFinalCost() const {
  auto Beg = InitialGraph->nodesBeg();
  auto End = InitialGraph->nodesEnd();
  auto TotalCost = Graph::Cost_t{0};
  auto NodePtrToIdx = std::map<const Graph::Node *, size_t>{};
  for (size_t Idx = 0; Beg != End; ++Idx, ++Beg)
    NodePtrToIdx[Beg->get()] = Idx;

  for (auto [NodePtr, Idx] : NodePtrToIdx) {
    assert(SelectedVariants.find(Idx) != SelectedVariants.end());
    auto Selection = SelectedVariants.find(Idx)->second;
    TotalCost += NodePtr->getCost(Selection);
  }

  auto GetSelection = 
    [&](const Graph::Node *NodePtr) {
      assert(NodePtrToIdx.find(NodePtr) != NodePtrToIdx.end());
      auto Idx = NodePtrToIdx.find(NodePtr)->second;
      assert(SelectedVariants.find(Idx) != SelectedVariants.end());
      return SelectedVariants.find(Idx)->second;
    };

  for (auto &Edge : utils::makeRange(InitialGraph->edgesBeg(), 
                                     InitialGraph->edgesEnd())) {
    auto [LhsNode, RhsNode] = Edge->getNodes();
    auto LhsSelection = GetSelection(LhsNode);
    auto RhsSelection = GetSelection(RhsNode);
    TotalCost += Edge->getCost(LhsSelection, RhsSelection);
  }

  if (FinalCost && !utils::isEqual(TotalCost, *FinalCost))
    utils::reportFatalError("Selections and final cost are not coherent:");
  return TotalCost;
}

void Solution::makeFinal(Graph InitialGraphIn) {
  if (isFinal())
    utils::reportFatalError("Solution is already final");
  resolveBoundedSolutions();
  InitialGraph = std::move(InitialGraphIn);
  auto NumOfUnresolvedNodes = 
    std::abs(static_cast<int>(InitialGraph->size() - SelectedVariants.size()));
  if (NumOfUnresolvedNodes != 0)
    utils::reportFatalError("There are " + 
      std::to_string(NumOfUnresolvedNodes) + " unresolved nodes");
  FinalCost = calcFinalCost();
}

void Solution::print(std::ostream &OS) const {
  if (!isFinal())
    utils::reportFatalError("Only final solution might be printed");

  OS << "graph Dump {\n"
     << "node[" << SolutionColour << "]\n";

  OS << "\"Solution:" << *FinalCost << "\" [" << AnswerNodeColour << "]\n";

  auto Beg = InitialGraph->nodesBeg();
  auto End = InitialGraph->nodesEnd();
  for (size_t Idx = 0; Beg != End; ++Idx, ++Beg) {
    assert(SelectedVariants.find(Idx) != SelectedVariants.end());
    OS << "\"" << Beg->get() << "\" [label = \"" << (*Beg)->getName() << " "
       << SelectedVariants.find(Idx)->second << "\"]\n";
  }

  for (auto &Edge :
       utils::makeRange(InitialGraph->edgesBeg(), InitialGraph->edgesEnd())) {
    assert(Edge);
    auto [Lhs, Rhs] = Edge->getNodes();
    OS << "\"" << Lhs << "\" -- \"" << Rhs << "\"\n";
  }
  OS << "}\n";
}

void Solution::printSummary(std::ostream &OS) const {
  OS << "Num of selections: " << SelectedVariants.size() << "\n";
  OS << "Selections:\n";
  for (auto [NodeIdx, Selection] : SelectedVariants)
    OS << "\t" << NodeIdx << " -> " << Selection << "\n";
}

struct LLVMNode final {
  std::vector<Graph::Cost_t> Cost;
  std::string Name;
  
  static std::regex getNodeRegex() {
    return std::regex("([[:digit:]])+ \\([_[:alnum:]]+:(%[[:digit:]]+)\\): \\[(.*)\\]");
  }

  static Graph::Cost_t dataToFloat(const std::string &Str) {
    if (Str == "INF")
      return std::numeric_limits<Graph::Cost_t>::infinity();
    return std::stof(Str);
  }

  static std::vector<Graph::Cost_t> parseData(std::string Data) {
    std::replace( Data.begin(), Data.end(), ',', ' ');
    auto SS = std::stringstream(Data);
    auto ParsedData = std::vector<Graph::Cost_t>{};
    auto Token = std::string{};
    while (std::getline(SS, Token))
      ParsedData.push_back(dataToFloat(Token));
  }

  static std::pair<size_t, LLVMNode> parse(std::string Str) {
    auto FullRegex = getNodeRegex();
    auto EndIt = std::sregex_token_iterator();

    auto NumIt = std::sregex_token_iterator(Str.cbegin(), Str.cend(), FullRegex, 1);
    assert (NumIt != EndIt);
    auto Num = std::stol(*NumIt);
    
    auto NameIt = std::sregex_token_iterator(Str.cbegin(), Str.cend(), FullRegex, 2);
    assert(NameIt != EndIt);

    auto DataIt = std::sregex_token_iterator(Str.cbegin(), Str.cend(), FullRegex, 3);
    assert(DataIt != EndIt);

    return {Num,LLVMNode{parseData(*DataIt), *NameIt}};
  }
};

// Function to read PBQP graph from LLVM representation
Graph GraphBuilders::readLLVM(std::istream &IS) {

}

} // namespace PBQP