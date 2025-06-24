#include "PBQP.h"

#include <fstream>

int main(int Argc, char **Argv) {
  auto CLPars = utils::CLParser{Argc, Argv};
  CLPars.addOption("in-file", utils::CLOption::Type::String);
  CLPars.addOption("out-file", utils::CLOption::Type::String);
  CLPars.addOption("LLVM", utils::CLOption::Type::Flag);

  CLPars.parseOptions();
  auto InFileName = std::string{""};
  auto OutFileName = std::string{"graph.dot"};
  auto ParseLLVM = false;

  if (CLPars.getOption("in-file") != "")
    InFileName = CLPars.getOption("in-file");
  if (CLPars.getOption("out-file") != "")
    OutFileName = CLPars.getOption("out-file");
  if (CLPars.getOption("LLVM") != "")
    ParseLLVM = CLPars.getOption("LLVM") == "true";

  if (InFileName.empty())
    utils::reportFatalError("Specify graph");

  auto IS = std::ifstream{InFileName};
  auto OS = std::ofstream{OutFileName};
  auto Graph = PBQP::Graph{};
  if (ParseLLVM) {
    Graph = PBQP::GraphBuilders::readLLVM(IS);
  } else {
    Graph.read(IS);
  }
  assert(Graph.validate());
  Graph.print(OS);
}