#include "PBQP.h"
#include "CPU-solver.h"
#include "GPU-solver.cu.h"

#include <fstream>
#include <iterator>

void printProfileInfo(PBQP::GPUSolver::PassManager::Profile_t &ProfileInfo) {
    std::cout << "Profile info:\n\t";
    std::transform(ProfileInfo.begin(), ProfileInfo.end(),
                   std::ostream_iterator<std::string>(std::cout, "\n\t"),
                   [](auto &Profile) {
                     return Profile.first + ": " + 
                            std::to_string(Profile.second);
                   });
    std::cout << std::endl;
}

size_t measureGPU(const std::string &InFileName, const std::string &AnsFileName,
                  bool OnlyTime) {
  auto IS = std::ifstream{InFileName};
  auto Graph = PBQP::Graph{};
  Graph.read(IS);
  assert(Graph.validate());
  auto Solver = PBQP::GPUFullSearch{};

  auto Start = std::chrono::steady_clock::now();
  auto Solution = Solver.solve(std::move(Graph));
  auto End = std::chrono::steady_clock::now();

  auto ProfileInfo = Solver.getProfileInfo();
  if (!OnlyTime)
    printProfileInfo(ProfileInfo);
  
  auto SolutionOS = std::ofstream{"GPU-" + AnsFileName};
  Solution.print(SolutionOS);
  return utils::to_milliseconds(End - Start);
}

size_t measureCPU(const std::string &InFileName, const std::string &AnsFileName) {
  auto IS = std::ifstream{InFileName};
  auto Graph = PBQP::Graph{};
  Graph.read(IS);
  assert(Graph.validate());
  auto Solver = PBQP::CPUFullSearch{};

  auto Start = std::chrono::steady_clock::now();
  auto Solution = Solver.solve(std::move(Graph));
  auto End = std::chrono::steady_clock::now();

  auto SolutionOS = std::ofstream{"CPU-" + AnsFileName};
  Solution.print(SolutionOS);
  return utils::to_milliseconds(End - Start);
}

void checkSolution(const std::string &InFileName) {
  auto IS = std::ifstream{InFileName};
  auto Graph = PBQP::Graph{};
  Graph.read(IS);
  assert(Graph.validate());
  auto CPUSolver = PBQP::CPUFullSearch{};
  auto GPUSolver = PBQP::GPUFullSearch{};

  auto CPUAns = CPUSolver.solve(PBQP::Graph::copy(Graph));
  auto GPUAns = GPUSolver.solve(PBQP::Graph::copy(Graph));

  if (!utils::isEqual(CPUAns.getFinalCost(), GPUAns.getFinalCost()))
    utils::reportFatalError("Differen answers: CPU[" + 
                            std::to_string(CPUAns.getFinalCost()) + "] GPU[" +
                            std::to_string(GPUAns.getFinalCost()) + "]");
}

void runHeuristic(const std::string &InFileName) {
  auto IS = std::ifstream{InFileName};
  auto Graph = PBQP::Graph{};
  Graph.read(IS);
  assert(Graph.validate());

  auto Solver = PBQP::HeuristicSolver{};
  Solver.solve(std::move(Graph));

  auto ProfileInfo = Solver.getProfileInfo();
  printProfileInfo(ProfileInfo);
}

int main(int Argc, char **Argv) {
  auto CLPars = utils::CLParser{Argc, Argv};
  CLPars.addOption("in-file", utils::CLOption::Type::String);
  CLPars.addOption("out-file", utils::CLOption::Type::String);
  CLPars.addOption("use-GPU", utils::CLOption::Type::Flag);
  CLPars.addOption("use-heuristic", utils::CLOption::Type::Flag);
  CLPars.addOption("use-CPU", utils::CLOption::Type::Flag);
  CLPars.addOption("only-time", utils::CLOption::Type::Flag);
  CLPars.addOption("check-solution", utils::CLOption::Type::Flag);

  CLPars.parseOptions();
  auto InFileName = std::string{"./graphs/default.out"};
  auto OutFileName = std::string{"solution.dot"};
  auto UseGPU = CLPars.getOption("use-GPU") == "true";
  auto UseCPU = CLPars.getOption("use-CPU") == "true";
  auto UseHeuristic = CLPars.getOption("use-heuristic") == "true";
  auto OnlyTime = false;
  auto CheckSolution = false;

  if (CLPars.getOption("in-file") != "")
    InFileName = CLPars.getOption("in-file");
  if (CLPars.getOption("out-file") != "")
    OutFileName = CLPars.getOption("out-file");
  if (CLPars.getOption("only-time") != "")
    OnlyTime = CLPars.getOption("only-time") == "true";
  if (CLPars.getOption("check-solution") != "")
    CheckSolution = CLPars.getOption("check-solution") == "true";

  if (UseCPU && UseGPU)
    utils::reportFatalError("Use only one solver");
  
  if (!UseCPU && !UseGPU && !UseHeuristic)
    utils::reportFatalError("No solver has been specified");

  auto OutString = std::string{};
  if (UseGPU) {
    auto Time = measureGPU(InFileName, OutFileName, OnlyTime);
    OutString = std::to_string(Time) + "\n";
    if (!OnlyTime)
      OutString = "GPU time: " + std::to_string(Time) + "ms\n";
  }

  if (UseCPU) {
    auto Time = measureCPU(InFileName, OutFileName);
    OutString = std::to_string(Time) + "\n";
    if (!OnlyTime)
      OutString = "CPU time: " + std::to_string(Time) + "ms\n";
  }

  if (UseHeuristic) {
    runHeuristic(InFileName);
  }

  std::cout << OutString << std::endl;

  if (CheckSolution)
    checkSolution(InFileName);
}