#include <iostream>
#include <chrono>
#include <thread>

extern __global__ void MatMul();

__device__ int A[1];

int main() {
  std::cout << "Cuda" << std::endl;
  //int *GlobA = nullptr;                     //this is device memory. so we can't print it
  int LocA = 2;
  //cudaMalloc(&GlobA, sizeof(GlobA));

  cudaMemcpyToSymbol(A, &LocA, sizeof(int));
  MatMul<<<10, 1>>>();
  cudaDeviceSynchronize();
  cudaMemcpyFromSymbol(&LocA, A, sizeof(int));

  std::cout << "A: " << LocA << std::endl;
  //cudaFree(GlobA);
  std::cout << "Cuda done"  << std::endl;
}