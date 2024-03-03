#include "PBQP.h"

#include <fstream>

int main() {
  auto Proxy = std::vector<float>{1, 2, 3, 4};
  auto Matrix = host::Matrix<float>{Proxy.begin(), Proxy.end(), 2, 2};
  auto Vector = host::Matrix<float>{Proxy.begin(), Proxy.begin() + 2, 2, 1};
  auto Graph = PBQP::Graph{};
  auto &Node1 = Graph.addNode(Vector);
  Vector[0][0] = 2;
  auto &Node2 = Graph.addNode(Vector);
  Vector[0][0] = 3;
  auto &Node3 = Graph.addNode(Vector);

  Graph.addEdge(Node1, Matrix, Node2);
  Matrix[0][0] = 5;
  Graph.addEdge(Node2, Matrix, Node3);
  Matrix[0][0] = 6;
  Graph.addEdge(Node3, Matrix, Node1);
  
  auto DotGraphOS = std::ofstream{"graph.dot"};
  auto GraphOS = std::ofstream{"graph.out"};
  Graph.print(DotGraphOS);
  Graph.dump(GraphOS);
  
  GraphOS.close();
  auto NewGraph = PBQP::Graph{};
  auto NewGraphIS = std::ifstream{"graph.out"};
  NewGraph.read(NewGraphIS);

  auto NewDotGraphOS = std::ofstream{"new-graph.dot"};
  NewGraph.print(NewDotGraphOS);
}