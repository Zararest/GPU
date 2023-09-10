#pragma once

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cmath>
#include <iostream>

#define DEBUG

#ifdef DEBUG
#define DEBUG_EXPR(expr) (expr)
#else 
#define DEBUG_EXPR(expr)
#endif

#define CUDA_CHECK(expr) {  auto MyErr = (expr) ;                                     \
                            if (MyErr != cudaSuccess) {                               \
                              printf("%s in %s at line %d\n",                         \
                                      cudaGetErrorString(MyErr), __FILE__, __LINE__); \
                              exit(1);                                                \
                            }}


struct HostMatrix;

HostMatrix generate(size_t Height, size_t Width);
HostMatrix referenceMul(HostMatrix &A, HostMatrix &B);
void printDeviceLimits(std::ostream &S);
void checkKernelsExec();

template <typename T1, typename T2>
size_t ceilDiv(T1 Lhs, T2 Rhs) {
  auto LhsF = static_cast<float>(Lhs);
  auto RhsF = static_cast<float>(Rhs);
  return std::ceil(LhsF / RhsF);
}