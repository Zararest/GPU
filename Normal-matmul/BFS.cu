#include "Kernels.cu.h"
#include "Matr-utils.h"

struct Config {
  bool Check = false;
  bool PrintOnlyTime = false;
  size_t Size = 32;
};

int main(int Argc, char **Argv) {
  auto Res = host::generateGraph(50, 2, 1);
  std::cout << Res.Graph.h() << "x" << Res.Graph.w() << std::endl;
  //print(Res.Graph, std::cout, std::boolalpha);
  std::cout << "\nBFS:" << std::endl;
  for (auto Level : Res.BFS)
    std::cout << Level << " ";
  std::cout << std::endl;
}