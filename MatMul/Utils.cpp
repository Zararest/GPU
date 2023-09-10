#include "Utils.h"

#include <random>

#define MAX_FLOAT 1000

HostMatrix generate(size_t Height, size_t Width) {
  std::mt19937 Rng(4);
  std::uniform_int_distribution<std::mt19937::result_type> Dist(1, MAX_FLOAT);

  auto Size = Height * Width;
  auto Res = HostMatrix{Height, Width};
  for (auto &It : Res.Elements)
    It = static_cast<float>(Dist(Rng)) / static_cast<float>(Dist(Rng));
  return Res;
}

HostMatrix referenceMul(HostMatrix &A, HostMatrix &B) {
  assert(A.Width == B.Height);
  auto NewMatrix = HostMatrix{A.Height, B.Width};
  for (size_t Row = 0; Row < A.Height; ++Row)
    for (size_t Col = 0; Col < B.Width; ++Col)
      for (size_t k = 0; k < A.Width; ++k)
        NewMatrix[Row][Col] += A[Row][k] * B[k][Col];
  return NewMatrix;
}   