#pragma once

#include "Matrix.h"
#include "Utils.h"

template <typename T>
using Tile_t = typename device::Matrix<T>::Tile; 

template <typename T>
__device__
void __fillTiles(Tile_t<T> &ATile, device::Matrix<T> &A,
               Tile_t<T> &BTile, device::Matrix<T> &B,
               size_t Iteration, size_t Size) {
  // A tile X position and B tile Y position are the same
  auto CurTilesPos = Size * Iteration;
  auto ATileYPos = Size * blockIdx.y;
  auto BTileXPos = Size * blockIdx.x;
  ATile[threadIdx.y][threadIdx.x] = 0.0;
  BTile[threadIdx.y][threadIdx.x] = 0.0;

  if (threadIdx.x + CurTilesPos < A.w() && 
      threadIdx.y + ATileYPos < A.h())
    ATile[threadIdx.y][threadIdx.x] = 
      A[ATileYPos + threadIdx.y][CurTilesPos + threadIdx.x];

  if (threadIdx.x + BTileXPos < B.w() && 
      threadIdx.y + CurTilesPos < B.h())
    BTile[threadIdx.y][threadIdx.x] = 
      B[CurTilesPos + threadIdx.y][BTileXPos + threadIdx.x];
}

template <typename T, int BlockSize>
__global__
void __tiledMatMul(device::Matrix<T> A, device::Matrix<T> B, 
                   device::Matrix<T> C) {

  __shared__ T AShMem[BlockSize * BlockSize];
  __shared__ T BShMem[BlockSize * BlockSize];
  auto ATile = Tile_t<T>{BlockSize, AShMem};
  auto BTile = Tile_t<T>{BlockSize, BShMem};
  // this should have T type 
  float Res = 0.0;
  for (size_t i = 0; i < utils::ceilDiv(A.w(), blockDim.x); ++i) {
    __fillTiles(ATile, A, BTile, B, i, BlockSize);
    __syncthreads();

    for (size_t k = 0; k < BlockSize; ++k)
      Res += ATile[threadIdx.y][k] * BTile[k][threadIdx.x];
    __syncthreads();
  }
  auto Col = blockIdx.x * blockDim.y + threadIdx.x;
  auto Row = blockIdx.y * blockDim.x + threadIdx.y;
  
  if (Row >= A.h() || Col >= B.w())
    return;
  
  C[Row][Col] = Res;
}

template <typename T, int BlockSize>
host::MatMulResult<T> tiledMatMul(const host::Matrix<T> &A, const host::Matrix<T> &B) {
  auto A_d = device::Matrix<T>{A};
  auto B_d = device::Matrix<T>{B};
  auto C_d = device::Matrix<T>{A.h(), B.w()};
  assert(A.w() == B.h());

  dim3 ThrBlockDim{BlockSize, BlockSize};
  dim3 BlockGridDim{utils::ceilDiv(B.w(), ThrBlockDim.x),
                    utils::ceilDiv(A.h(), ThrBlockDim.y)};

  auto Start = std::chrono::steady_clock::now();
  __tiledMatMul<T, BlockSize><<<BlockGridDim, ThrBlockDim>>>(A_d, B_d, C_d);
  cudaDeviceSynchronize();
  utils::checkKernelsExec();
  auto End = std::chrono::steady_clock::now();

  auto Res = 
    host::MatMulResult<T>{C_d.getHostMatrix(), 
      std::chrono::duration_cast<std::chrono::milliseconds>(End - Start).count()};
  A_d.free();
  B_d.free();
  C_d.free();
  return Res;
}

template <typename T>
__device__
void __fillSharedVector(T *VShared, device::Matrix<T> &V, 
                        size_t Iteration, size_t Size) {
  VShared[threadIdx.x] = V[0][Iteration * Size + threadIdx.x];
}

template <typename T, int BlockSize>
__global__
void __vectMatrKernel(device::Matrix<T> V, device::Matrix<T> A, 
                      device::Matrix<T> C) {
  __shared__ T VShMem[BlockSize];

  T Res = 0.0;
  for (size_t i = 0; i < utils::ceilDiv(V.w(), BlockSize); ++i) {
    __fillSharedVector(VShMem, V, i, BlockSize);
    __syncthreads();

    for (size_t k = 0; k < BlockSize; ++k)
      if (i * BlockSize + k < A.h() && blockIdx.x * BlockSize + threadIdx.x < A.w())
        Res += VShMem[k] * A[i * BlockSize + k][blockIdx.x * BlockSize + threadIdx.x];
    __syncthreads();
  }

  auto Col = blockIdx.x * BlockSize + threadIdx.x;
  
  if (Col >= C.w())
    return;
  
  C[0][Col] = Res;
}

template <typename T, int BlockSize>
host::MatMulResult<T> vectMatrMul(const host::Matrix<T> &V, const host::Matrix<T> &A) {
  assert(V.h() == 1);
  assert(V.w() == A.h());
  auto V_d = device::Matrix<T>{V};
  auto A_d = device::Matrix<T>{A};
  auto C_d = device::Matrix<T>{V.h(), A.w()};

  dim3 ThrBlockDim{BlockSize};
  dim3 BlockGridDim{utils::ceilDiv(A.h(), ThrBlockDim.x)};

  auto Start = std::chrono::steady_clock::now();
  __vectMatrKernel<T, BlockSize><<<BlockGridDim, ThrBlockDim>>>(V_d, A_d, C_d);
  cudaDeviceSynchronize();
  utils::checkKernelsExec();
  auto End = std::chrono::steady_clock::now();

  auto Res = 
    host::MatMulResult<T>{C_d.getHostMatrix(), 
      std::chrono::duration_cast<std::chrono::milliseconds>(End - Start).count()};
  V_d.free();
  A_d.free();
  C_d.free();
  return Res;
}

template <typename T, int BlockSize>
host::MatMulResult<T> MatrVectMul(const host::Matrix<T> &A, const host::Matrix<T> &V) {
  auto Res = 
    vectMatrMul<T, BlockSize>(host::Matrix<T>::transpose(V), host::Matrix<T>::transpose(A));
  Res.Matr = host::Matrix<T>::transpose(Res.Matr);
  return Res;
}

template <typename T, int BlockSize>
host::MatMulResult<T> optimizedMatMul(const host::Matrix<T> &A, const host::Matrix<T> &B) {
  if (A.h() == 1)
    return vectMatrMul<T, BlockSize>(A, B);
  if (B.w() == 1)
    return MatrVectMul<T, BlockSize>(A, B);
  return tiledMatMul<T, BlockSize>(A, B);
}