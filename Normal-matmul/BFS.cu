#include "Kernels.cu.h"
#include "Matr-utils.h"

#include <fstream>
#include <algorithm>
#include <numeric>

enum class Device {
  CPU,
  GPU,
  GPUNoCopy
};

struct Config {
  bool Check = false;
  bool PrintOnlyTime = false;
  bool OnlyGenerate = false;
  std::string FileToRead;
  std::string FileToDump;
  size_t Size = 32;
  Device DeviceToCalc = Device::GPU;
};

constexpr size_t BlockSize = 16;

template <Device Type>
std::vector<size_t> calculateBFS(host::Matrix<host::Relation> &Graph) {
  utils::reportFatalError("unknown type");
  return {};
}

template <typename Inserter>
void getNeighbors(size_t NodeNum, host::Matrix<host::Relation> &Graph, Inserter It) {
  for (size_t DestNode = 0; DestNode  < Graph.w(); ++DestNode)
    if (Graph[NodeNum][DestNode] != 0)
      It = DestNode;
}

template <>
std::vector<size_t> calculateBFS<Device::CPU>(host::Matrix<host::Relation> &Graph) {
  assert(Graph.h() == Graph.w());
  auto NodesNum = Graph.h();
  auto Ans = std::vector<size_t>(NodesNum, 0ull);
  auto CurNodes = std::set<size_t>{};
  auto NewNodes = std::set<size_t>{};
  auto VisitedNodes = std::set<size_t>{};
  auto CurLevel = 1ull;

  CurNodes.insert(0ull);
  VisitedNodes.insert(0ull);
  while (VisitedNodes.size() < NodesNum) {
    NewNodes.clear();
    for (auto Node : CurNodes)
      getNeighbors(Node, Graph, std::inserter(NewNodes, NewNodes.begin()));  
    for (auto VisitedNode : VisitedNodes)
      NewNodes.erase(VisitedNode);
    
    if (NewNodes.size() == 0)
      utils::reportFatalError("The graph is disconnected");

    for (auto NewNode : NewNodes) {
      Ans[NewNode] = CurLevel;
      VisitedNodes.insert(NewNode);
    }

    CurNodes = std::move(NewNodes);
    CurLevel++;
  }
  return Ans;
}

template <typename Matr_t>
std::vector<size_t> __calculateBFS(Matr_t &Graph) {
  assert(Graph.h() == Graph.w());
  auto NodesNum = Graph.h();
  auto Ans = std::vector<size_t>(NodesNum, 0ull);
  auto Mask = std::vector<host::Relation>(NodesNum, 1ull);
  auto CurNodes = host::Matrix<host::Relation>{1, NodesNum};
  std::fill(CurNodes.begin(), CurNodes.end(), 0ul);
  auto CurLevel = 1ull;
  auto VisitedNodesNum = 1ull;
  
  auto KernelTime = 0ull;
  Mask[0] = 0; // root node already visited
  CurNodes[0][0] = 1; // initial node is 0
  while (VisitedNodesNum < NodesNum) {
    auto Start = std::chrono::steady_clock::now();
    auto MatMulRes = optimizedMatMul<BlockSize>(CurNodes, Graph);
    auto End = std::chrono::steady_clock::now();
    KernelTime += std::chrono::duration_cast<std::chrono::milliseconds>(End - Start).count();

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
// here [a][b] = 1 means that a->b exists
// this means that multiplication should be like this:
//   (nodes)x(Graph)
template <>
std::vector<size_t> calculateBFS<Device::GPU>(host::Matrix<host::Relation> &Graph) {
  return __calculateBFS(Graph);
}

template <>
std::vector<size_t> calculateBFS<Device::GPUNoCopy>(host::Matrix<host::Relation> &Graph) {
  auto Graph_d = device::Matrix<host::Relation>{Graph};
  auto Res = __calculateBFS(Graph_d);
  Graph_d.free();
  return Res;
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
  
  auto Start = std::chrono::steady_clock::now();

  auto CalculatedBFS = std::vector<size_t>{};
  switch (Cfg.DeviceToCalc) {
  case Device::GPU:
    CalculatedBFS = calculateBFS<Device::GPU>(Graph);
    break;
  case Device::CPU:
    CalculatedBFS = calculateBFS<Device::CPU>(Graph);
    break;
  case Device::GPUNoCopy:
    CalculatedBFS = calculateBFS<Device::GPUNoCopy>(Graph);
    break;
  default:
    utils::reportFatalError("Unknown type");
  };

  auto End = std::chrono::steady_clock::now();
  auto FullDuration =
    std::chrono::duration_cast<std::chrono::milliseconds>(End - Start).count();
  
  if (Cfg.PrintOnlyTime) {
    std::cout << FullDuration;
  } else {
    std::cout << "BFS duration: " << FullDuration << "ms" << std::endl;
  }

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

    if (Option == "--CPU") {
      BFSConfig.DeviceToCalc = Device::CPU;
      continue;
    }

    if (Option == "--GPU") {
      BFSConfig.DeviceToCalc = Device::GPU;
      continue;
    }

    if (Option == "--GPUNoCopy") {
      BFSConfig.DeviceToCalc = Device::GPUNoCopy;
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