#include <cuda_runtime.h>
#include <iostream>

__constant__ int Diff = 10;
extern __device__ int A[1];

__global__ void MatMul() { 
  atomicAdd(A, 1); //надо использовать атомик потому что в общем случае это УБ
  printf("In kernel: %i\n", A[0]);
}