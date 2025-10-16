#include "GPU-solver.cu.h"

#include <fstream>
#include <iostream>
#include <string>

void runSolver(std::string InputName, std::string OutName) {
  auto IS = std::ifstream{InputName};
  auto OS = std::ofstream{OutName};
  auto Graph = PBQP::readGraph(IS, /*ParseLLVM*/ true);
  auto Solver = PBQP::ReductionsSolver{};

  auto Start = std::chrono::steady_clock::now();
  auto Solution = Solver.solve(std::move(Graph));
  auto End = std::chrono::steady_clock::now();

  auto ProfileInfo = Solver.getProfileInfo();
  auto ExecutionTime =
    utils::to_milliseconds_fractional(utils::to_microseconds(End - Start));
  auto TimeWithoutLoader = ExecutionTime -
    utils::to_milliseconds_fractional(ProfileInfo[0]);
  OS << "Total cost: " << Solution.getFinalCost() << "\n";
  OS << "Execution time: " << ExecutionTime << "\n";
  OS << "Pure execution time: " << TimeWithoutLoader << "\n";
}

int main(int Argc, char* Argv[]) {
  // Check command line arguments
  if (Argc != 2) {
    std::cerr << "Usage: " << Argv[0] << " <input_file.pbqpgraph>" << std::endl;
    return 1;
  }

  std::string Filename = Argv[1], Extension = ".pbqpgraph";

  // Check file extension
  if (Filename.find(Extension) == std::string::npos) {
      std::cerr << "Warning: Input file does not have .pbqpgraph extension" << std::endl;
  }

  std::string OutName = Filename;
  OutName.erase(OutName.find(Extension, 0), Extension.length());
  OutName += ".solution-gpu";

  std::cout << "Starting GPU solver for file: " << Filename << std::endl;
  runSolver(Filename, OutName);

  return 0;
}