#include "Kernels.cu.h"
#include "Matr-utils.h"

#include <memory>

enum class MulType {
  Tiled,
  CPU,
  CPUOpt
};

constexpr size_t BlockSize = 16;

struct Config {
  bool CheckMat = false;
  bool PrintOnFail = false;
  bool PrintOnlyTime = false;
  bool TimeWithoutCopy = false;
  bool Print = false;
  MulType Type = MulType::Tiled;
  size_t Heigth = 5, Width = 3, JointSize = 2;
};

template <typename T>
void matMul(Config MatrConfig) {
  using MatMulRes = typename host::MatMulResult<T>;

  auto A = host::generate<T>(MatrConfig.Heigth, MatrConfig.JointSize);
  auto B = host::generate<T>(MatrConfig.JointSize, MatrConfig.Width);

  auto Start = std::chrono::steady_clock::now();
  auto Res = MatMulRes{host::Matrix<T>{1, 1}, 0};

  switch (MatrConfig.Type) {
  case MulType::CPU:
    Res = host::matMul<T>(A, B);
    break;

  case MulType::Tiled:
    Res = tiledMatMul<T, BlockSize>(A, B);
    break;

  default:
    assert(false && "Unknown type");
    break;
  }

  auto End = std::chrono::steady_clock::now();
  auto FullDuration =
    std::chrono::duration_cast<std::chrono::milliseconds>(End - Start);

  auto Duration = MatrConfig.TimeWithoutCopy ? Res.DurationMs :
                                               FullDuration.count();
  
  if (MatrConfig.CheckMat && !check(A, B, Res.Matr)) {
    std::cout << "Wrong answer" << std::endl;
    if (MatrConfig.PrintOnFail) {
      std::cout << "Real result:\n";
      check(A, B, Res.Matr, /*DumpOnFail*/ true);
      std::cout << "\n\nMy result:\n";
      host::print(Res.Matr, std::cout);
    }
    return;
  }

  if (MatrConfig.PrintOnlyTime) {
    std::cout << Duration;
    return;
  }

  std::cout << "Multiplication time:" << Duration << "ms" << std::endl;
}

int main(int Argc, char **Argv) {
  Argv++;
  Argc--;
  auto MatrConfig = Config{};
  while (Argc > 0) {
    auto Option = std::string{Argv[0]};
    Argv++;
    Argc--;
    if (Option == "--check") {
      MatrConfig.CheckMat = true;
      continue;
    }

    if (Option == "--check-with-dump") {
      MatrConfig.PrintOnFail = true;
      MatrConfig.CheckMat = true;
      continue;
    }

    if (Option == "--only-time") {
      MatrConfig.PrintOnlyTime = true;
      continue;
    }

    if (Option == "--matrix") {
      if (Argc < 3)
        utils::reportFatalError("Too few arguments");
      MatrConfig.Heigth = std::stoi(Argv[0]);
      MatrConfig.Width = std::stoi(Argv[1]);
      MatrConfig.JointSize = std::stoi(Argv[2]);
      Argv += 3;
      Argc -= 3;
      continue;
    }

    if (Option == "--params") {
      utils::printDeviceLimits(std::cout);
      continue;
    }

    if (Option == "--tiled") {
      MatrConfig.Type = MulType::Tiled;
      continue;
    }

    if (Option == "--CPU") {
      MatrConfig.Type = MulType::CPU;
      continue;
    }

    utils::reportFatalError("Unknown argument: " + Option);
  }

  if (!MatrConfig.PrintOnlyTime) {
    std::cout << "Matricies sizes: A{" << MatrConfig.Heigth << ", "
              << MatrConfig.JointSize << "} B{" << MatrConfig.JointSize
              << ", " << MatrConfig.Width << "}" << std::endl;
    std::cout << "Mult type: ";
    switch (MatrConfig.Type) {
    case MulType::CPU:
      std::cout << "on CPU" << std::endl;
      break;
    
    case MulType::Tiled:
      std::cout << "tiled on GPU" << std::endl;
      break;
    
    default:
      assert(false && "Unknown type");
    } 
  }

  matMul<float>(MatrConfig);
}