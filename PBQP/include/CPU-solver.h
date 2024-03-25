#pragma once

#include "PBQP.h"

#include <unordered_map>

namespace PBQP {

class CPUFullSearch final : public Solver {
  Graph::Cost_t MinCost;
  std::unique_ptr<Solution> OptimalSolution;
  std::unordered_map<Graph::Node *, size_t> NodePtrToIdx;

  void updateSolution(const std::vector<size_t> &Selections, 
                      Graph::Cost_t CurCost) {
    MinCost = CurCost;
    OptimalSolution->clear();
    for (size_t Idx = 0; Idx < Selections.size(); ++Idx)
      OptimalSolution->addSelection(Idx, Selections[Idx]);
  }

  void checkSolution(const std::vector<size_t> &Selections) {
    auto &Graph = OptimalSolution->getGraph();
    auto CurCost = Graph::Cost_t{0};
    for (auto &Edge : utils::makeRange(Graph.edgesBeg(), Graph.edgesEnd())) {
      auto [LhsAddr, RhsAddr] = Edge->getNodes();
      assert(NodePtrToIdx.find(LhsAddr) != NodePtrToIdx.end() && 
             NodePtrToIdx[LhsAddr] < Selections.size());
      assert(NodePtrToIdx.find(RhsAddr) != NodePtrToIdx.end() && 
             NodePtrToIdx[RhsAddr] < Selections.size());
      auto LhsSelecton = Selections[NodePtrToIdx[LhsAddr]];
      auto RhsSelecton = Selections[NodePtrToIdx[RhsAddr]];
      CurCost += Edge->getCost(LhsSelecton, RhsSelecton);
      if (CurCost == Graph::InfCost)
        return;
    }

    auto NodesBeg = Graph.nodesBeg();
    auto NodesEnd = Graph.nodesEnd();
    for (size_t Idx = 0; Idx < Selections.size(); ++NodesBeg, ++Idx) {
      assert(NodesBeg != NodesEnd);
      CurCost += (*NodesBeg)->getCost(Selections[Idx]);
    }
    if (MinCost > CurCost)
      updateSolution(Selections, CurCost);
  }

  void checkAllOptions(std::vector<size_t> &Selections) {
    auto CurNodeIdx = Selections.size();
    if (CurNodeIdx >= OptimalSolution->getGraph().size()) {
      checkSolution(Selections);
      return;
    }
    
    auto CurNodeIt = OptimalSolution->getGraph().nodesBeg();
    std::advance(CurNodeIt, CurNodeIdx);
    auto &CurNodeCost = (*CurNodeIt)->getCostVector();
    Selections.push_back(0);
    for (size_t i = 0; i < (*CurNodeIt)->costSize(); ++i) {
      Selections.back() = i;
      checkAllOptions(Selections);
    }
    Selections.pop_back();
  }

  static std::unordered_map<Graph::Node *, size_t> createNodesCache(Graph &Task) {
    auto NodesBeg = Task.nodesBeg();
    auto NodesEnd = Task.nodesEnd();
    auto NodePtrToIdx = std::unordered_map<Graph::Node *, size_t>{};
    for (size_t i = 0; NodesBeg != NodesEnd; ++NodesBeg, ++i)
      NodePtrToIdx[NodesBeg->get()] = i;
    return NodePtrToIdx;
  } 

public:
  Solution solve(Graph Task) override {
    NodePtrToIdx = createNodesCache(Task);
    OptimalSolution = std::make_unique<Solution>(std::move(Task));
    MinCost = Graph::InfCost;
    auto Buf = std::vector<size_t>{};
    checkAllOptions(Buf);
    OptimalSolution->addFinalCost(MinCost);
    return std::move(*OptimalSolution.release());
  }
};

} // namespace PBQP 