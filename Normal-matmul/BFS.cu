#include "Kernels.cu.h"
#include "Matr-utils.h"

#include <fstream>

struct Config {
  bool Check = false;
  bool PrintOnlyTime = false;
  size_t Size = 32;
};

int main(int Argc, char **Argv) {
  auto Res = host::generateGraph(10, 2, 1);
  auto FileName = "Matr-dump";
  auto OFStream = std::ofstream{FileName};
  host::dumpMatrix(Res.Graph, OFStream);

  auto IFStream = std::ifstream{FileName};
  auto FileMatr = host::readMatrix<host::Relation>(IFStream);
  host::dumpMatrix<host::Relation>(FileMatr, std::cout);
}