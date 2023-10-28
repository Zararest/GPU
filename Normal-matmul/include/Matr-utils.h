#pragma once

#include "Matrix.h"
#include "Utils.h"

#include <random>
#include <set>
#include <iterator>

namespace host {

constexpr size_t MaxInt = 10;
constexpr size_t MaxFloat = 10;
constexpr size_t Seed = 1;

template <typename T>
Matrix<T> generate(size_t Height, size_t Width);

template <>
Matrix<float> generate(size_t Height, size_t Width) {
  std::mt19937 Rng(Seed);
  std::uniform_real_distribution<> Dist(0.1, MaxFloat);

  auto Res = Matrix<float>{Height, Width};
  for (auto &It : utils::makeRange(Res.begin(), Res.end()))
    It = Dist(Rng);
  return Res;
}

template <>
Matrix<int> generate(size_t Height, size_t Width) {
  std::mt19937 Rng(Seed);
  std::uniform_int_distribution<> Dist(0, MaxInt);

  auto Res = Matrix<int>{Height, Width};
  for (auto &It : utils::makeRange(Res.begin(), Res.end()))
    It = Dist(Rng);
  return Res;
}

template <typename T>
MatMulResult<T> matMul(Matrix<T> &A, Matrix<T> &B) {
  auto NewMatrix = Matrix<T>{A.h(), B.w()};
  auto Start = std::chrono::steady_clock::now();
  for (size_t Row = 0; Row < A.h(); ++Row)
    for (size_t Col = 0; Col < B.w(); ++Col)
      for (size_t k = 0; k < A.w(); ++k)
        NewMatrix[Row][Col] += A[Row][k] * B[k][Col];
  auto End = std::chrono::steady_clock::now();

  return MatMulResult<T>{NewMatrix, 
      std::chrono::duration_cast<std::chrono::milliseconds>(End - Start).count()};
}

template <typename T>
void print(Matrix<T> Matr, std::ostream &S) {
  for (size_t y = 0; y < Matr.w() ; ++y) {
    for (size_t x = 0; x < Matr.h(); ++x)
      S << Matr[y][x] << " ";
    S << "\n";
  }
}

template <typename T>
bool check(Matrix<T> &A, Matrix<T> &B, Matrix<T> &Res, bool DumpOnFail = false) {
  auto RealRes = matMul<T>(A, B).Matr;
  if (RealRes.w() != Res.w() || RealRes.h() != Res.h())
    return false;
  if (!std::equal(RealRes.begin(), RealRes.end(), Res.begin(),  
                  [](const T &Lhs, const T &Rhs) {
                    auto e = Lhs * 0.01;
                    return Rhs > Lhs - e && Rhs < Lhs + e;
                  })) {
    print(RealRes, std::cout);
    return false;
  }
  return true;
}

using Relation = unsigned char;

struct GraphGenRes {
  Matrix<unsigned char> Graph;
  std::vector<size_t> BFS;
};

struct Edge {
  size_t From;
  size_t To;

  Edge(size_t From, size_t To) : From{From}, To{To} {}
};

size_t chooseNode(const std::set<size_t> Nodes, 
                  std::uniform_int_distribution<size_t> &NeibDist, 
                  std::mt19937 &Rng) {
  auto NodePos = NeibDist(Rng);
  auto NodeIt = Nodes.begin();
  std::advance(NodeIt, NodePos % Nodes.size());
  assert(NodeIt != Nodes.end());
  return *NodeIt;
}

std::vector<Edge> generateNewLevel(const std::set<size_t> &Visited, 
                      const std::set<size_t> &NotVisited,
                      const std::set<size_t> &CurrentlyVisited,
                      size_t CurNumOfNeib, size_t CurNumOfVisitiees,
                      std::uniform_int_distribution<size_t> &NeibDist, 
                      std::mt19937 &Rng) {
  assert(CurNumOfVisitiees <= CurNumOfNeib);
  auto NewEdges = std::vector<Edge>{};
  for (size_t i = 0; i < CurNumOfVisitiees; ++i) {
    auto From = chooseNode(CurrentlyVisited, NeibDist, Rng);
    auto To = chooseNode(NotVisited, NeibDist, Rng);
    NewEdges.emplace_back(From, To);
  }

  for (size_t i = 0; i < (CurNumOfNeib - CurNumOfVisitiees); ++i) {
    auto From = chooseNode(CurrentlyVisited, NeibDist, Rng);
    auto To = chooseNode(Visited, NeibDist, Rng);
    NewEdges.emplace_back(From, To);
  }
  return NewEdges;
}

template <typename It>
void fillGraph(Matrix<unsigned char> &Graph, It EdgesBeg, It EdgesEnd) {
  for (auto Edge : utils::makeRange(EdgesBeg, EdgesEnd)) {
    assert(!Graph[Edge.From][Edge.To]);
    Graph[Edge.From][Edge.To] = true;
  }
}

template <typename It>
void fillVisitSets(std::set<size_t> &Visited, 
                   std::set<size_t> &NotVisited, 
                   std::set<size_t> &CurrentlyVisited,
                   It EdgesBeg, It EdgesEnd) {
  auto NewCurVisited = std::set<size_t>{};
  for (auto Edge : utils::makeRange(EdgesBeg, EdgesEnd))
    if (NotVisited.find(Edge.To) != NotVisited.end())
      NewCurVisited.insert(Edge.From);

  for (auto Node : CurrentlyVisited) {
    auto Inserted = Visited.insert(Node);
    assert(Inserted.second);
    NotVisited.erase(Node);
  }

  CurrentlyVisited = std::move(NewCurVisited);
}

// Graph - matrix NxN
// Graph[a][b] == true  =>  graph has a->b 
GraphGenRes generateGraph(size_t Size, double AverageNeighboursNum, 
                          double AverageBFSVisiteesNum) {
  auto BFS = std::vector<size_t>(Size);
  auto Graph = Matrix<unsigned char>{Size, Size};
  auto Visited = std::set<size_t>{};
  auto NotVisited = std::set<size_t>{};
  auto Rng = std::mt19937(Seed);
  auto NeibNumDist = 
    std::normal_distribution<>{AverageNeighboursNum, AverageNeighboursNum / 2};
  auto VisiteesNum = 
    std::normal_distribution<>{AverageBFSVisiteesNum, 
                             AverageBFSVisiteesNum / 2};
  auto NeibDist = std::uniform_int_distribution<size_t>{0, Size};

  // I don't know better way to do this
  for (size_t NodeNum = 0; NodeNum < Size; ++NodeNum)
    NotVisited.emplace(NodeNum);
  std::fill(Graph.begin(), Graph.end(), 0);  
  std::fill(Graph.begin(), Graph.end(), 0);

  constexpr auto Root = 0ul;
  auto CurrentlyVisited = std::set<size_t>{Root};
  auto Level = 1ul;
  while (NotVisited.size() > 0) {
    auto CurNumOfNeib = 
      static_cast<size_t>(std::ceil(std::abs(NeibNumDist(Rng))));
    auto CurNumOfVisitiees = 
      static_cast<size_t>(std::ceil(std::abs(VisiteesNum(Rng)))) % CurNumOfNeib;
    CurNumOfVisitiees = CurNumOfVisitiees ? CurNumOfVisitiees : 1;
    auto NewEdges = generateNewLevel(Visited, NotVisited, CurrentlyVisited,
                                     CurNumOfNeib, CurNumOfVisitiees,
                                     NeibDist, Rng);

    fillGraph(Graph, NewEdges.begin(), NewEdges.end());
    fillVisitSets(Visited, NotVisited, CurrentlyVisited, 
                  NewEdges.begin(), NewEdges.end());
    assert(Visited.size() + NotVisited.size() == Size);

    for (auto Node : CurrentlyVisited)
      BFS[Node] = Level;
  }
  return {Graph, BFS};
}
}// namespace host