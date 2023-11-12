#include "Kernels.cu.h"
#include "Matr-utils.h"

#include <fstream>
#include <algorithm>
#include <numeric>

struct Config {
  bool Check = false;
  bool PrintOnlyTime = false;
  bool OnlyGenerate = false;
  std::string FileToRead;
  std::string FileToDump;
  size_t Size = 32;
};

constexpr size_t BlockSize = 16;

// here [a][b] = 1 means that a->b exists
// this means that we should transpose matrixes
std::vector<size_t> calculateBFS(host::Matrix<host::Relation> &Graph) {
  assert(Graph.h() == Graph.w());
  auto NodesNum = Graph.h();
  auto Ans = std::vector<size_t>(NodesNum, 0ull);
  auto Mask = std::vector<host::Relation>(NodesNum, 1ull);
  auto CurNodes = host::Matrix<host::Relation>{1, NodesNum};
  std::fill(CurNodes.begin(), CurNodes.end(), 0ul);
  auto CurLevel = 1ull;
  auto VisitedNodesNum = 1ull;

  Mask[0] = 0; // root node already visited
  CurNodes[0][0] = 1; // initial node is 0
  while (VisitedNodesNum < NodesNum) {
    auto MatMulRes = tiledMatMul<host::Relation, BlockSize>(CurNodes, Graph);
    // unmasked new nodes
    auto NewNodes = std::move(MatMulRes.Matr);
    // mask visited nodes
    auto MaskedNodes = host::Matrix<host::Relation>{1, NodesNum};
    std::transform(NewNodes.begin(), NewNodes.end(), 
                   Mask.begin(), 
                   MaskedNodes.begin(), 
                   [](host::Relation NewNode, size_t Mask) {
                     // return value should be 0 or 1, 
                     // but NewNode may be greater than 1
                     return NewNode && Mask;
                   });
    auto NewNodesNum = std::accumulate(MaskedNodes.begin(), MaskedNodes.end(), 0ull);
    if (NewNodesNum == 0)
      utils::reportFatalError("The graph is disconnected");
    VisitedNodesNum += NewNodesNum;
    // adding levels to the answer
    std::transform(Ans.begin(), Ans.end(), 
                   MaskedNodes.begin(), 
                   Ans.begin(),
                   [CurLevel](size_t AnsElem, host::Relation NewNode) {
                    DEBUG_EXPR(assert((NewNode == 1 && AnsElem == 0) ||
                                       NewNode == 0));
                    return AnsElem + NewNode * CurLevel;
                   });
    std::transform(MaskedNodes.begin(), MaskedNodes.end(), 
                   Mask.begin(),
                   Mask.begin(),
                   [](host::Relation NewNode, size_t Mask) {
                    DEBUG_EXPR(assert(NewNode <= 1));
                    DEBUG_EXPR(assert(Mask <= 1));
                    return !NewNode && Mask;
                   });
    CurNodes = std::move(MaskedNodes);
    CurLevel++;
  }

  return Ans;
}

void BFS(Config Cfg) {
  auto Ans = std::vector<size_t>{};
  auto Graph = host::Matrix<host::Relation>{};
  if (!Cfg.FileToRead.empty()) {
    auto S = std::ifstream{Cfg.FileToRead};
    if (!S.is_open())
      utils::reportFatalError("Can't open file to read");
    Ans = host::readBFS(S);
    Graph = host::readMatrix<host::Relation>(S);
  } else {
    auto GenRes = host::generateGraph(Cfg.Size);
    Ans = std::move(GenRes.BFS);
    Graph = std::move(GenRes.Graph);
  }

  auto CalculatedBFS = calculateBFS(Graph);

  if (Cfg.Check && (CalculatedBFS.size() != Ans.size() || 
                    !std::equal(CalculatedBFS.begin(), CalculatedBFS.end(),
                                Ans.begin()))) {
    if (!Cfg.FileToDump.empty()) {
      auto S = std::ofstream{Cfg.FileToDump};
      if (!S.is_open())
        utils::reportFatalError("Can't open file to dump wrong answer");
      host::dumpBFS(CalculatedBFS, S);
    }
    utils::reportFatalError("Wrong BFS");
  }
    
  
  if (!Cfg.FileToDump.empty()) {
    auto S = std::ofstream{Cfg.FileToDump};
    if (!S.is_open())
      utils::reportFatalError("Can't open file to write");
    host::dumpBFS(Ans, S);
    host::dumpMatrix(Graph, S);
  }
}

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
    if (BFSConfig.FileToRead.empty())
      std::cout << "Matrix size: " << BFSConfig.Size << std::endl;

    if (BFSConfig.OnlyGenerate && BFSConfig.FileToDump.empty())
      utils::reportFatalError("Empty dump file");

    if (BFSConfig.OnlyGenerate) {
      auto GenRes = host::generateGraph(BFSConfig.Size);
      auto S = std::ofstream{BFSConfig.FileToDump};
      if (!S.is_open())
        utils::reportFatalError("Can't open file to dump generated graph");
      host::dumpBFS(GenRes.BFS, S);
      host::dumpMatrix(GenRes.Graph, S);
      return 0;
    }
  } 

  BFS(BFSConfig);
}