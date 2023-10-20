#include "Matrix.h"
#include "Utils.h"

#include <chrono>

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
  
  float Res = 0.0;
  for (size_t i = 0; i < ceilDiv(A.w(), blockDim.x); ++i) {
    __fillTiles(ATile, A, BTile, B, i, BlockSize);
    __syncthreads();

    for (size_t i = 0; i < BlockSize; ++i)
      Res += ATile[threadIdx.y][i] * BTile[i][threadIdx.x];
    __syncthreads();
  }
  auto Col = blockIdx.x * blockDim.y + threadIdx.x;
  auto Row = blockIdx.y * blockDim.x + threadIdx.y;
  
  if (Row >= A.h() || Col >= B.w())
    return;
  
  C[Row][Col] = Res;
}

template <typename T>
struct MatMulResult {
  host::Matrix<T> Matr;
  std::chrono::duration<long, std::milli> Duration;
};

template <typename T, int BlockSize>
MatMulResult<T> tiledMatMul(const host::Matrix<T> &A, const host::Matrix<T> &B) {
  auto A_d = device::Matrix<T>{A};
  auto B_d = device::Matrix<T>{B};
  auto C_d = device::Matrix<T>{A.h(), B.w()};
  assert(A.w() == B.h());

  dim3 ThrBlockDim{BlockSize, BlockSize};
  dim3 BlockGridDim{ceilDiv(B.w(), ThrBlockDim.x),
                    ceilDiv(A.h(), ThrBlockDim.y)};

  auto Start = std::chrono::steady_clock::now();
  __tiledMatMul<T, BlockSize><<<BlockGridDim, ThrBlockDim>>>(A_d, B_d, C_d);
  cudaDeviceSynchronize();
  checkKernelsExec();
  auto End = std::chrono::steady_clock::now();

  auto Res = 
    MatMulResult<T>{C_d.getHostMatrix(), 
      std::chrono::duration_cast<std::chrono::milliseconds>(End - Start)};
  A_d.free();
  B_d.free();
  C_d.free();
  return Res;
} 