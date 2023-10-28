#include "Kernels.cu.h"
#include "Matr-utils.h"

#include <fstream>

struct Config {
  bool Check = false;
  bool PrintOnlyTime = false;
  bool OnlyGenerate = false;
  std::string FileToRead;
  std::string FileToDump;
  size_t Size = 32;
};

int main(int Argc, char **Argv) {
  Argv++;
  Argc--;
  auto BFSConfig = Config{};
  while (Argc > 0) {
    auto Option = std::string{Argv[0]};
    Argv++;
    Argc--;
    if (Option == "--check") {
      BFSConfig.Check = true;
      continue;
    }

    if (Option == "--only-time") {
      BFSConfig.PrintOnlyTime = true;
      continue;
    }

    if (Option == "--size") {
      if (Argc < 1)
        utils::reportFatalError("Too few arguments");
      BFSConfig.Size = std::stoi(Argv[0]);
      Argv++;
      Argc--;
      continue;
    }

    if (Option == "--only-gen") {
      BFSConfig.OnlyGenerate = true;
      continue;
    }

    if (Option == "--input") {
      if (Argc < 1)
        utils::reportFatalError("Too few arguments");
      BFSConfig.FileToRead = Argv[0];
      Argv++;
      Argc--;
      continue;
    }

    if (Option == "--output") {
      if (Argc < 1)
        utils::reportFatalError("Too few arguments");
      BFSConfig.FileToDump = Argv[0];
      Argv++;
      Argc--;
      continue;
    }

    utils::reportFatalError("Unknown argument: " + Option);
  }

  if (!BFSConfig.PrintOnlyTime) {
    std::cout << "Matrix size: " << BFSConfig.Size << std::endl;
    if (BFSConfig.OnlyGenerate && BFSConfig.FileToDump.empty())
      utils::reportFatalError("Empty dump file");

    if (BFSConfig.OnlyGenerate) {
      auto GenRes = host::generateGraph(BFSConfig.Size);
      auto S = std::ofstream{BFSConfig.FileToDump};
      host::dumpBFS(GenRes.BFS, S);
      host::dumpMatrix(GenRes.Graph, S);
    }
  } 
}