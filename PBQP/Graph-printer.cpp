#include "PBQP.h"

#include <fstream>

int main(int Argc, char **Argv) {
  auto CLPars = utils::CLParser{Argc, Argv};
  CLPars.addOption("in-file", utils::CLOption::Type::String);
  CLPars.addOption("out-file", utils::CLOption::Type::String);

  CLPars.parseOptions();
  auto InFileName = std::string{""};
  auto OutFileName = std::string{"graph.dot"};

  if (CLPars.getOption("in-file") != "")
    InFileName = CLPars.getOption("in-file");
  if (CLPars.getOption("out-file") != "")
    OutFileName = CLPars.getOption("out-file");
  
  if (InFileName.empty())
    utils::reportFatalError("Specify graph");
  
  auto IS = std::ifstream{InFileName};
  auto OS = std::ofstream{OutFileName};
  auto Graph = PBQP::Graph{};
  Graph.read(IS);
  assert(Graph.validate());
  Graph.print(OS);
}