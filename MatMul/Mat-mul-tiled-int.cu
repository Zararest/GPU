#include <chrono>
#include <iostream>

#include "Matrix.h"
#include "Utils.h"

__device__ void fillTiles(size_t Iteration, TileInt &ATile, DevMatRowMajorInt A,
                          TileInt &BTile, DevMatRowMajorInt B) {
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

__global__ void tiledMatMulKernel(DevMatRowMajorInt A, DevMatRowMajorInt B,
                                  DevMatRowMajorInt C) {
  constexpr auto TileWidth = BlockSize;
  assert(TileWidth == blockDim.x);
  assert(TileWidth == blockDim.y);
  auto Col = blockIdx.x * blockDim.y + threadIdx.x;
  auto Row = blockIdx.y * blockDim.x + threadIdx.y;

  auto NumOfTiles = ceilDiv(A.Width, blockDim.x);
  __shared__ int ASharedMem[TileWidth * TileWidth];
  __shared__ int BSharedMem[TileWidth * TileWidth];
  auto ATile = TileInt{TileWidth, /*X*/ 0u, blockIdx.y * TileWidth, ASharedMem};
  auto BTile = TileInt{TileWidth, blockIdx.x * TileWidth, /*Y*/ 0u, BSharedMem};

  int Res = 0.0;
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

HostMatrixInt tiledMatMulInt(const HostMatrixInt &A, const HostMatrixInt &B, bool PrintTime) {
  DevMatRowMajorInt DevA{A};
  DevMatRowMajorInt DevB{B};
  DevMatRowMajorInt DevC{A.Height, B.Width};
  assert(DevMatRowMajorInt::checkMul(DevA, DevB, DevC));

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
  auto Res = DevMatRowMajorInt::getHostMat(DevC);
  DevA.free();
  DevB.free();
  DevC.free();
  return Res;
}