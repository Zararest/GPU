#pragma once

#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <set>

#ifdef DEBUG
#define DEBUG_EXPR(expr) (expr)
#else
#define DEBUG_EXPR(expr)
#endif

#define CUDA_CHECK(expr)                                                       \
  {                                                                            \
    auto MyErr = (expr);                                                       \
    if (MyErr != cudaSuccess) {                                                \
      printf("%s in %s at line %d\n", cudaGetErrorString(MyErr), __FILE__,     \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  }

namespace utils {

void printDeviceLimits(std::ostream &S);
void checkKernelsExec();

template <typename T1, typename T2>
__device__ __host__ unsigned ceilDiv(T1 Lhs, T2 Rhs) {
  auto LhsF = static_cast<float>(Lhs);
  auto RhsF = static_cast<float>(Rhs);
  return ceil(LhsF / RhsF);
}

template <typename It>
class IteratorRange {
  It Begin;
  It End;

public:
  IteratorRange(It Begin, It End) : Begin{Begin}, End{End} {}

  It begin() { return Begin; }
  It end() { return End; }
};

template <typename It>
IteratorRange<It> makeRange(It Begin, It End) {
  return IteratorRange<It>{Begin, End};
}

template <typename It>
void print(It Beg, It End, std::ostream &S) {
  for (auto I : makeRange(Beg, End))
    S << I << " ";
  S << "\n";
}

template <typename T>
std::set<T> sub(const std::set<size_t> &Lhs, const std::set<T> &Rhs) {
  auto Ans = Lhs;
  for (auto I : Rhs)
    Ans.erase(I);
  return Ans;
}

void reportFatalError(const std::string &Msg);

} // namespace utils