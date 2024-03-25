#include "Graph-gen.h"

#include <fstream>

int main(int Argc, char **Argv) {
  auto CLPars = utils::CLParser{Argc, Argv};
  CLPars.addOption("nodes-num", utils::CLOption::Type::String);
  CLPars.addOption("num-of-cliques", utils::CLOption::Type::String);
  CLPars.addOption("node-size", utils::CLOption::Type::String);
  CLPars.addOption("avg-neighb-num", utils::CLOption::Type::String);
  CLPars.addOption("has-inf-cost", utils::CLOption::Type::Flag);
  CLPars.addOption("out-file", utils::CLOption::Type::String);

  CLPars.parseOptions();
  auto Cfg = PBQP::GenConfig{};
  auto OutFileName = std::string{"generated.out"};

  if (CLPars.getOption("nodes-num") != "")
    Cfg.NumOfNodes = std::stoi(CLPars.getOption("nodes-num"));
  if (CLPars.getOption("num-of-cliques") != "")
    Cfg.NumOfCliques = std::stoi(CLPars.getOption("num-of-cliques"));
  if (CLPars.getOption("node-size") != "")
    Cfg.VectSize = std::stoi(CLPars.getOption("node-size"));
  if (CLPars.getOption("avg-neighb-num") != "")
    Cfg.AvgNeighbNum = std::stod(CLPars.getOption("avg-neighb-num"));
  if (CLPars.getOption("has-inf-cost") != "")
    Cfg.AvgNeighbNum = CLPars.getOption("has-inf-cost") == "true";
  if (CLPars.getOption("out-file") != "")
    OutFileName = CLPars.getOption("out-file");

  auto Graph = PBQP::generateGraph(Cfg);
  auto GraphOS = std::ofstream{OutFileName};
  Graph.dump(GraphOS);
  assert(Graph.validate());
}