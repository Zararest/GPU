#include <chrono>
#include <iostream>

#include "Matrix.h"
#include "Utils.h"

int main() {
  auto A = generate(5, 2);
  auto B = generate(2, 2);
  std::cout << "A:" << std::endl;
  A.print(std::cout);
  std::cout << "B:" << std::endl;
  B.print(std::cout);

  auto Start = std::chrono::steady_clock::now();
  auto C = simpleMatMul(A, B);
  auto End = std::chrono::steady_clock::now();
  auto Duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(End - Start);
  std::cout << "Without shared: " << Duration.count() << std::endl;

  auto RealC = referenceMul(A, B);
  if (C.Elements.size() != RealC.Elements.size() ||
      !std::equal(C.Elements.begin(), C.Elements.end(), RealC.Elements.begin(),
                  [](float Lhs, float Rhs) {
                    auto e = 0.001;
                    return Rhs > Lhs - e && Rhs < Lhs + e;
                  })) {
    std::cout << "Wrong answer" << std::endl;
    std::cout << "Real:" << std::endl;
    RealC.print(std::cout);
    std::cout << "My:" << std::endl;
    C.print(std::cout);
  }

  /*Start = std::chrono::steady_clock::now();
  withShared();
  End = std::chrono::steady_clock::now();
  Duration = std::chrono::duration_cast<std::chrono::milliseconds>(End - Start);
  std::cout << "With shared: " << Duration.count() << std::endl;*/
}