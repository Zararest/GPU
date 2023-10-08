#include <chrono>
#include <iostream>

#include "Matrix.h"
#include "Utils.h"

enum class MulType { All, WithoutShared, Tiled, TransposeTiled };

struct Config {
  bool CheckMat = false;
  bool Print = false;
  MulType Type = MulType::All;
  size_t Heigth = 5, Width = 3, JointSize = 2;
};

void matMul(Config MatrConfig) {
  auto A = generate(MatrConfig.Heigth, MatrConfig.JointSize);
  auto B = generate(MatrConfig.JointSize, MatrConfig.Width);

  auto Start = std::chrono::steady_clock::now();
  auto C = HostMatrix{1, 1};
  auto TypeStr = std::string{"Unknown type"};

  switch (MatrConfig.Type) {
  case MulType::WithoutShared:
    C = simpleMatMul(A, B);
    TypeStr = "Without shared";
    break;

  case MulType::Tiled:
    C = tiledMatMul(A, B);
    TypeStr = "Tiled matrix";
    break;

  case MulType::TransposeTiled:
    C = transposeTiledMatMul(A, B);
    TypeStr = "Transpose tiled matrix";
    break;

  default:
    assert(false && "Unknown type");
    break;
  }

  auto End = std::chrono::steady_clock::now();
  auto Duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(End - Start);
  std::cout << TypeStr << ": " << Duration.count() << "ms" << std::endl;

  if (MatrConfig.Print) {
    std::cout << "A:" << std::endl;
    A.print(std::cout);
    std::cout << "B:" << std::endl;
    B.print(std::cout);
    std::cout << "C: " << std::endl;
    C.print(std::cout);
  }

  if (!MatrConfig.CheckMat)
    return;

  Start = std::chrono::steady_clock::now();
  auto RealC = referenceMul(A, B);
  End = std::chrono::steady_clock::now();
  Duration = std::chrono::duration_cast<std::chrono::milliseconds>(End - Start);
  std::cout << "CPU mult duration: " << Duration.count() << "ms" << std::endl;
  if (C.Elements.size() != RealC.Elements.size() ||
      !std::equal(C.Elements.begin(), C.Elements.end(), RealC.Elements.begin(),
                  [](float Lhs, float Rhs) {
                    auto e = Lhs * 0.0001;
                    if (Rhs <= Lhs - e || Rhs >= Lhs + e)
                      std::cout << "Differ: " << Lhs << " vs " << Rhs
                                << "\n\tdiff: " << Lhs - Rhs << std::endl;
                    return Rhs > Lhs - e && Rhs < Lhs + e;
                  })) {
    std::cout << "Wrong answer" << std::endl;
    if (!MatrConfig.Print)
      return;
    std::cout << "Real:" << std::endl;
    RealC.print(std::cout);
    std::cout << "My:" << std::endl;
    C.print(std::cout);
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
      std::cout << "Running with answer checker" << std::endl;
      continue;
    }

    if (Option == "--matrix") {
      assert(Argc >= 3 && "Too few arguments");
      MatrConfig.Heigth = std::stoi(Argv[0]);
      MatrConfig.Width = std::stoi(Argv[1]);
      MatrConfig.JointSize = std::stoi(Argv[2]);
      Argv += 3;
      Argc -= 3;
      std::cout << "Matricies sizes: A{" << MatrConfig.Heigth << ", "
                << MatrConfig.JointSize << "} B{" << MatrConfig.JointSize
                << ", " << MatrConfig.Width << "}" << std::endl;
      continue;
    }

    if (Option == "--print") {
      MatrConfig.Print = true;
      std::cout << "Running with matricies dump" << std::endl;
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

    if (Option == "--simple") {
      MatrConfig.Type = MulType::WithoutShared;
      continue;
    }

    if (Option == "--transposed") {
      MatrConfig.Type = MulType::TransposeTiled;
      continue;
    }

    std::cout << "Unknown argument: " << Option << std::endl;
    assert(false);
  }

  if (MatrConfig.Type == MulType::All) {
    MatrConfig.Type = MulType::WithoutShared;
    matMul(MatrConfig);
    MatrConfig.Type = MulType::Tiled;
    matMul(MatrConfig);
    MatrConfig.Type = MulType::TransposeTiled;
    matMul(MatrConfig);
    return 0;
  }

  matMul(MatrConfig);
}