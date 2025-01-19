#include "CPU-solver.h"
#include "GPU-solver.cu.h"
#include "PBQP.h"

#include <fstream>

void printProfileInfo(PBQP::GPUSolver::PassManager::Profile_t &ProfileInfo) {
  std::cout << "Profile info:\n\t";
  std::transform(ProfileInfo.begin(), ProfileInfo.end(),
                 std::ostream_iterator<std::string>(std::cout, "\n\t"),
                 [](auto &Profile) {
                   if (Profile.first.find("Loop header") != std::string::npos)
                     return Profile.first + ": " + std::to_string(Profile.second);
                   if (Profile.first.find("Loop end") != std::string::npos)
                     return Profile.first + "\n";
                   auto Duration = utils::to_milliseconds_fractional(Profile.second);
                   return Profile.first + ": " + std::to_string(Duration) + "ms";
                 });
  std::cout << std::endl;
}

double measureGPU(const std::string &InFileName, const std::string &AnsFileName,
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

  auto TimeWithoutLoad = ProfileInfo[1].second;

  auto SolutionOS = std::ofstream(AnsFileName + "-GPU.dot");
  assert(SolutionOS.is_open());
  Solution.print(SolutionOS);
  return utils::to_milliseconds_fractional(TimeWithoutLoad);
}

double measureCPU(const std::string &InFileName,
                  const std::string &AnsFileName) {
  auto IS = std::ifstream{InFileName};
  auto Graph = PBQP::Graph{};
  Graph.read(IS);
  assert(Graph.validate());
  auto Solver = PBQP::CPUFullSearch{};

  auto Start = std::chrono::steady_clock::now();
  auto Solution = Solver.solve(std::move(Graph));
  auto End = std::chrono::steady_clock::now();

  auto SolutionOS = std::ofstream(AnsFileName + "-CPU.dot");
  assert(SolutionOS.is_open());
  Solution.print(SolutionOS);
  return utils::to_milliseconds_fractional(utils::to_microseconds(End - Start));
}

void checkSolution(const std::string &InFileName) {
  auto IS = std::ifstream{InFileName};
  auto Graph = PBQP::Graph{};
  Graph.read(IS);
  assert(Graph.validate());
  auto GPUSolver = PBQP::ReductionsSolver{};
  auto RefSolver = PBQP::GPUFullSearch{};

  auto RefAns = RefSolver.solve(PBQP::Graph::copy(Graph));
  auto GPUAns = GPUSolver.solve(PBQP::Graph::copy(Graph));

  if (!utils::isEqual(RefAns.getFinalCost(), GPUAns.getFinalCost()))
    utils::reportFatalError("Differen answers: Ref[" +
                            std::to_string(RefAns.getFinalCost()) + "] GPU[" +
                            std::to_string(GPUAns.getFinalCost()) + "]");
}

double measureReductions(const std::string &InFileName,
                  const std::string &AnsFileName) {
  auto IS = std::ifstream{InFileName};
  auto Graph = PBQP::Graph{};
  Graph.read(IS);
  assert(Graph.validate());

  auto Solver = PBQP::ReductionsSolver{};

  auto Start = std::chrono::steady_clock::now();
  auto Solution = Solver.solve(std::move(Graph));
  auto End = std::chrono::steady_clock::now();

  auto ProfileInfo = Solver.getProfileInfo();
  printProfileInfo(ProfileInfo);

  auto SolutionOS = std::ofstream{AnsFileName + "-reductions.dot"};
  assert(SolutionOS.is_open());
  Solution.print(SolutionOS);
  return utils::to_milliseconds_fractional(utils::to_microseconds(End - Start));
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
    auto Time = measureReductions(InFileName, OutFileName);
    OutString = std::to_string(Time) + "\n";
    if (!OnlyTime)
      OutString = "Reductions on GPU time: " + std::to_string(Time) + "ms\n";
  }

  std::cout << OutString << std::endl;

  if (CheckSolution)
    checkSolution(InFileName);
}