#pragma once

#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>


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

struct HostMatrix;
struct HostMatrixInt;

HostMatrixInt generateInt(size_t Height, size_t Width);
HostMatrix generate(size_t Height, size_t Width);
HostMatrix referenceMul(HostMatrix &A, HostMatrix &B);
void printDeviceLimits(std::ostream &S);
void checkKernelsExec();

template <typename T1, typename T2>
__device__ __host__ size_t ceilDiv(T1 Lhs, T2 Rhs) {
  auto LhsF = static_cast<float>(Lhs);
  auto RhsF = static_cast<float>(Rhs);
  return ceil(LhsF / RhsF);
}