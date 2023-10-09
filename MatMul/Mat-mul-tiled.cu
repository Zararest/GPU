#include <chrono>
#include <iostream>

#include "Matrix.h"
#include "Utils.h"

__device__ void fillTiles(size_t Iteration, Tile &ATile, DevMatRowMajor A,
                          Tile &BTile, DevMatRowMajor B) {
  DEBUG_EXPR(assert(ATile.Size == BTile.Size));
  auto Size = ATile.Size;
  auto CurTilePos = Iteration * Size;
  ATile.X = CurTilePos;
  BTile.Y = CurTilePos;
  // this needs to omit check in tile calc
  ATile.get(threadIdx.y, threadIdx.x) = 0.0;
  BTile.get(threadIdx.y, threadIdx.x) = 0.0;
  if (threadIdx.x + ATile.X < A.Width && threadIdx.y + ATile.Y < A.Height)
    ATile.get(threadIdx.y, threadIdx.x) =
        A.get(ATile.Y + threadIdx.y, ATile.X + threadIdx.x);

  if (threadIdx.x + BTile.X < B.Width && threadIdx.y + BTile.Y < B.Height)
    BTile.get(threadIdx.y, threadIdx.x) =
        B.get(BTile.Y + threadIdx.y, BTile.X + threadIdx.x);
}

__global__ void tiledMatMulKernel(DevMatRowMajor A, DevMatRowMajor B,
                                  DevMatRowMajor C) {
  constexpr auto TileWidth = BlockSize;
  assert(TileWidth == blockDim.x);
  assert(TileWidth == blockDim.y);
  auto Col = blockIdx.x * blockDim.y + threadIdx.x;
  auto Row = blockIdx.y * blockDim.x + threadIdx.y;

  auto NumOfTiles = ceilDiv(A.Width, blockDim.x);
  __shared__ float ASharedMem[TileWidth * TileWidth];
  __shared__ float BSharedMem[TileWidth * TileWidth];
  auto ATile = Tile{TileWidth, /*X*/ 0u, blockIdx.y * TileWidth, ASharedMem};
  auto BTile = Tile{TileWidth, blockIdx.x * TileWidth, /*Y*/ 0u, BSharedMem};

  float Res = 0.0;
  for (size_t i = 0; i < NumOfTiles; ++i) {
    fillTiles(i, ATile, A, BTile, B);
    __syncthreads();

    for (size_t i = 0; i < TileWidth; ++i)
      Res += ATile.get(threadIdx.y, i) * BTile.get(i, threadIdx.x);
    __syncthreads();
  }

  if (Row >= A.Height || Col >= B.Width)
    return;

  C.get(Row, Col) = Res;
}

HostMatrix tiledMatMul(const HostMatrix &A, const HostMatrix &B, bool PrintTime) {
  DevMatRowMajor DevA{A};
  DevMatRowMajor DevB{B};
  DevMatRowMajor DevC{A.Height, B.Width};
  assert(DevMatRowMajor::checkMul(DevA, DevB, DevC));

  // tile size equals BlockSize
  dim3 ThrBlockDim{BlockSize, BlockSize};
  dim3 BlockGridDim{ceilDiv(B.Width, ThrBlockDim.x),
                    ceilDiv(A.Height, ThrBlockDim.y)};

  auto Start = std::chrono::steady_clock::now();
  tiledMatMulKernel<<<BlockGridDim, ThrBlockDim>>>(DevA, DevB, DevC);
  cudaDeviceSynchronize();
  checkKernelsExec();
  auto End = std::chrono::steady_clock::now();
  auto Duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(End - Start);
  DEBUG_EXPR(std::cout << "\tKernel duration: " << Duration.count() << "ms"
                       << std::endl);
  if (PrintTime)
    std::cout << Duration.count();

  DEBUG_EXPR(std::cout << "Tiled finished" << std::endl);
  auto Res = DevMatRowMajor::getHostMat(DevC);
  DevA.free();
  DevB.free();
  DevC.free();
  return Res;
}