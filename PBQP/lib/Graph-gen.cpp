#include "Graph-gen.h"

#include <cmath>
#include <random>
#include <set>

namespace PBQP {

namespace {

host::Matrix<Graph::Cost_t> genMatrix(size_t H, size_t W, bool HasInfCosts,
                                      std::mt19937 &Gen) {
  constexpr Graph::Cost_t MaxCost = 50;
  constexpr auto InfProb = 0.1;
  auto ValDist = std::uniform_real_distribution<>(0, MaxCost);
  auto InfDist = std::discrete_distribution<>{1 - InfProb, InfProb};
  auto Matrix = host::Matrix<Graph::Cost_t>{H, W};
  std::generate(Matrix.begin(), Matrix.end(),
                [&ValDist, &Gen]() { return ValDist(Gen); });
  if (HasInfCosts)
    std::transform(Matrix.begin(), Matrix.end(), Matrix.begin(),
                   [&InfDist, &Gen](Graph::Cost_t Cost) {
                     if (InfDist(Gen))
                       return Graph::InfCost;
                     return Cost;
                   });
  return Matrix;
}

class EdgesInfo final {
  size_t NumOfNodes;
  size_t CurNodeIdx;

  std::vector<size_t> NewNeighbors;

public:
  EdgesInfo(size_t NumOfNodes, size_t CurNodeIdx)
      : NumOfNodes{NumOfNodes}, CurNodeIdx{CurNodeIdx} {}

  void addNeighbors(double AvgNeighbNum, std::mt19937 &Gen) {
    if (AvgNeighbNum <= 0)
      return;
    auto NeighbSet = std::set<size_t>{};
    constexpr auto AvgNeighbDiv = 1.0;
    auto NeighbNumDistrib =
        std::normal_distribution{AvgNeighbNum, AvgNeighbDiv};
    auto NeighbDistrib =
        std::uniform_int_distribution<size_t>{0ull, NumOfNodes - 1};
    std::generate_n(std::inserter(NeighbSet, NeighbSet.begin()),
                    NeighbNumDistrib(Gen),
                    [&NeighbDistrib, &Gen] { return NeighbDistrib(Gen); });
    // Cur node shouldn't have itself as heighbor
    NeighbSet.erase(CurNodeIdx);
    // We've already added next index as heighbor
    NeighbSet.erase(CurNodeIdx + 1);
    if (CurNodeIdx != 0)
      NeighbSet.erase(CurNodeIdx - 1);
    NewNeighbors.clear();
    std::copy(NeighbSet.begin(), NeighbSet.end(),
              std::back_inserter(NewNeighbors));
  }

  size_t getCurNodeIdx() const { return CurNodeIdx; }

  auto neighbBegin() { return NewNeighbors.begin(); }

  auto neighbEnd() { return NewNeighbors.end(); }
};

void generateEdge(Graph &Clique, size_t LhsIdx, size_t RhsIdx, bool HasInfCosts,
                  std::mt19937 &Gen) {
  auto &Lhs = *(Clique.nodesBeg() + LhsIdx);
  auto &Rhs = *(Clique.nodesBeg() + RhsIdx);
  auto Matrix = genMatrix(Lhs->costSize(), Rhs->costSize(), HasInfCosts, Gen);
  Clique.addEdge(*Lhs, std::move(Matrix), *Rhs);
}

template <typename It>
void addEdgesByInfo(Graph &Clique, It Beg, It End, bool HasInfCosts,
                    std::mt19937 &Gen) {
  for (auto &Info : utils::makeRange(Beg, End)) {
    auto CurNodeIdx = Info.getCurNodeIdx();
    std::for_each(Info.neighbBegin(), Info.neighbEnd(), [&](size_t NeibIdx) {
      generateEdge(Clique, CurNodeIdx, NeibIdx, HasInfCosts, Gen);
    });
  }
}

void addEdges(Graph &Clique, size_t VectSize, double AvgNeighbNum,
              bool HasInfCosts, std::mt19937 &Gen) {
  auto NodeIdexes = std::vector<size_t>(Clique.size());
  std::iota(NodeIdexes.begin(), NodeIdexes.end(), 0);
  auto InfoForNodes = std::vector<EdgesInfo>{};
  std::transform(NodeIdexes.begin(), NodeIdexes.end(),
                 std::back_inserter(InfoForNodes),
                 [NumOfNodes = Clique.size()](size_t Idx) {
                   return EdgesInfo{NumOfNodes, Idx};
                 });
  std::for_each(InfoForNodes.begin(), InfoForNodes.end(),
                [AvgNeighbNum, &Gen](EdgesInfo &Info) {
                  Info.addNeighbors(AvgNeighbNum, Gen);
                });
  addEdgesByInfo(Clique, InfoForNodes.begin(), InfoForNodes.end(), HasInfCosts,
                 Gen);
}

Graph generateClique(size_t NumOfNodes, size_t VectSize, double AvgNeighbNum,
                     bool HasInfCosts, std::mt19937 &Gen,
                     std::string CliquePrefix) {
  auto CostVectors = std::vector<host::Matrix<Graph::Cost_t>>(NumOfNodes);
  auto CostMatrices = std::vector<host::Matrix<Graph::Cost_t>>(NumOfNodes - 1);
  std::generate(CostVectors.begin(), CostVectors.end(), [VectSize, &Gen]() {
    return genMatrix(VectSize, 1, /*HasInfCosts*/ false, Gen);
  });
  std::generate(CostMatrices.begin(), CostMatrices.end(),
                [VectSize, HasInfCosts, &Gen]() {
                  return genMatrix(VectSize, VectSize, HasInfCosts, Gen);
                });

  auto Clique = Graph{};
  auto Idx = 0ull;
  std::for_each(
      CostVectors.begin(), CostVectors.end(),
      [&Clique, &Idx, CliquePrefix](host::Matrix<Graph::Cost_t> &Cost) {
        auto &Node = Clique.addNode(std::move(Cost));
        Node.changeName(CliquePrefix + "::" + std::to_string(Idx));
        Idx++;
      });

  for (auto NodeIdx = 0; NodeIdx < NumOfNodes - 1; ++NodeIdx) {
    auto LhNodeIt = Clique.nodesBeg();
    std::advance(LhNodeIt, NodeIdx);
    auto &LhsNode = *LhNodeIt;
    auto &RhsNode = *std::next(LhNodeIt);
    Clique.addEdge(*LhsNode, std::move(CostMatrices[NodeIdx]), *RhsNode);
  }

  // We have already generated 1 heighbor
  AvgNeighbNum--;
  addEdges(Clique, VectSize, AvgNeighbNum, HasInfCosts, Gen);
  assert(Clique.validate());
  return Clique;
}

} // anonymous namespace

Graph generateGraph(GenConfig Cfg) {
  std::random_device Rd;
  std::mt19937 Gen(Rd());
  auto CliqueSizesBounds = std::vector<size_t>{};
  // Last bound is NumOfNodes
  std::generate_n(std::back_inserter(CliqueSizesBounds), Cfg.NumOfCliques - 1,
                  [Field = Cfg.NumOfNodes, &Gen] { return Gen() % Field; });
  std::sort(CliqueSizesBounds.begin(), CliqueSizesBounds.end());
  CliqueSizesBounds.push_back(Cfg.NumOfNodes);
  CliqueSizesBounds.insert(CliqueSizesBounds.begin(), 0u);

  auto CliqueSizes = std::vector<size_t>{};
  std::transform(CliqueSizesBounds.begin(), std::prev(CliqueSizesBounds.end()),
                 std::next(CliqueSizesBounds.begin()),
                 std::back_inserter(CliqueSizes),
                 [](size_t LhsBound, size_t RhsBound) {
                   assert(RhsBound >= LhsBound);
                   return std::max(RhsBound - LhsBound, 1ul);
                 });

  auto CliqueIdxes = std::vector<size_t>(CliqueSizes.size());
  std::iota(CliqueIdxes.begin(), CliqueIdxes.end(), 0);
  auto Cliques = std::vector<Graph>{};
  std::transform(
      CliqueSizes.begin(), CliqueSizes.end(), CliqueIdxes.begin(),
      std::back_inserter(Cliques), [&Gen, &Cfg](size_t CliqueSize, size_t Idx) {
        return generateClique(CliqueSize, Cfg.VectSize, Cfg.AvgNeighbNum,
                              Cfg.HasInfCosts, Gen, std::to_string(Idx));
      });
  auto FinalGraph = Graph{};
  std::for_each(Cliques.begin(), Cliques.end(),
                [&FinalGraph](const Graph &Clique) {
                  FinalGraph = Graph::merge(FinalGraph, Clique);
                });
  return FinalGraph;
}

} // namespace PBQP