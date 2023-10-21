#pragma once

#include "Matrix.h"
#include "Utils.h"

#include <random>

namespace host {

constexpr size_t MaxInt = 10;
constexpr size_t MaxFloat = 10;
constexpr size_t Seed = 1;

template <typename T>
Matrix<T> generate(size_t Height, size_t Width);

template <>
Matrix<float> generate(size_t Height, size_t Width) {
  std::mt19937 Rng(Seed);
  std::uniform_real_distribution<> Dist(0.1, MaxFloat);

  auto Res = Matrix<float>{Height, Width};
  for (auto &It : makeRange(Res.begin(), Res.end()))
    It = Dist(Rng);
  return Res;
}

template <>
Matrix<int> generate(size_t Height, size_t Width) {
  std::mt19937 Rng(Seed);
  std::uniform_int_distribution<> Dist(0, MaxInt);

  auto Res = Matrix<int>{Height, Width};
  for (auto &It : makeRange(Res.begin(), Res.end()))
    It = Dist(Rng);
  return Res;
}

template <typename T>
MatMulResult<T> matMul(Matrix<T> &A, Matrix<T> &B) {
  auto NewMatrix = Matrix<T>{A.h(), B.w()};
  auto Start = std::chrono::steady_clock::now();
  for (size_t Row = 0; Row < A.h(); ++Row)
    for (size_t Col = 0; Col < B.w(); ++Col)
      for (size_t k = 0; k < A.w(); ++k)
        NewMatrix[Row][Col] += A[Row][k] * B[k][Col];
  auto End = std::chrono::steady_clock::now();

  return MatMulResult<T>{NewMatrix, 
      std::chrono::duration_cast<std::chrono::milliseconds>(End - Start)};
}

}// namespace host