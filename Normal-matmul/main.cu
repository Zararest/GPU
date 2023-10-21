#include "Kernels.cu.h"
#include "Generator.h"

#include <memory>

enum class MulType {
  Tiled,
  CPU
};

constexpr size_t BlockSize = 16;

struct Config {
  bool CheckMat = false;
  bool PrintOnlyTime = false;
  MulType Type = MulType::Tiled;
  size_t Heigth = 5, Width = 3, JointSize = 2;
};

template <typename T>
void matMul(Config MatrConfig) {
  using MatMulRes = typename host::MatMulResult<T>;

  auto A = host::generate<T>(MatrConfig.Heigth, MatrConfig.JointSize);
  auto B = host::generate<T>(MatrConfig.JointSize, MatrConfig.Width);

  auto Start = std::chrono::steady_clock::now();
  auto Res = std::unique_ptr<MatMulRes>(nullptr);
  auto TypeStr = std::string{"Unknown type"};

  switch (MatrConfig.Type) {
  case MulType::CPU:
    Res = std::unique_ptr<MatMulRes>(host::matMul<T>(A, B));
    TypeStr = "Without shared";
    break;

  case MulType::Tiled:
    Res = std::unique_ptr<MatMulRes>(tiledMatMul<T, BlockSize>(A, B));
    TypeStr = "Tiled matrix";
    break;

  default:
    assert(false && "Unknown type");
    break;
  }
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

    if (Option == "--only-time") {
      MatrConfig.PrintOnlyTime = true;
      continue;
    }

    if (Option == "--matrix") {
      assert(Argc >= 3 && "Too few arguments");
      MatrConfig.Heigth = std::stoi(Argv[0]);
      MatrConfig.Width = std::stoi(Argv[1]);
      MatrConfig.JointSize = std::stoi(Argv[2]);
      Argv += 3;
      Argc -= 3;
      continue;
    }

    if (Option == "--params") {
      printDeviceLimits(std::cout);
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

    std::cout << "Unknown argument: " << Option << std::endl;
    assert(false);
  }


  if (!MatrConfig.PrintOnlyTime) {
    std::cout << "Matricies sizes: A{" << MatrConfig.Heigth << ", "
              << MatrConfig.JointSize << "} B{" << MatrConfig.JointSize
              << ", " << MatrConfig.Width << "}" << std::endl;
    std::cout << "Mult type: ";
    switch (MatrConfig.Type) {
    case MulType::CPU:
      std::cout << "on CPU";
      break;
    
    case MulType::Tiled:
      std::cout << "tiled on GPU";
      break;
    
    default:
      assert(false && "Unknown type");
    } 
  }

  matMul<float>(MatrConfig);
}