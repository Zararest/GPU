#include "Kernels.cu.h"

int main() {
  auto A = host::Matrix<float>{1, 1};
  auto B = host::Matrix<float>{1, 1};

  tiledMatMul<float, 16>(A, B);
}