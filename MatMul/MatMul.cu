#include <chrono>
#include <iostream>

#include "Matrix.h"
#include "Utils.h"

// Thread block size
constexpr size_t BlockSize = 16;

__global__ void simpleMatMulKernel(DeviceMatrix A, DeviceMatrix B,
                                   DeviceMatrix C) {
  auto Col = blockIdx.x * blockDim.y + threadIdx.x;
  auto Row = blockIdx.y * blockDim.x + threadIdx.y;

  if (Row >= A.Height || Col >= B.Width)
    return;

  float Res = 0;
  for (size_t i = 0; i < A.Width; ++i)
    Res += A.get(Row, i) * B.get(i, Col);
  C.get(Row, Col) = Res;
}

HostMatrix simpleMatMul(const HostMatrix &A, const HostMatrix &B) {
  DeviceMatrix DevA{A};
  DeviceMatrix DevB{B};
  DeviceMatrix DevC{A.Height, B.Width};
  assert(DeviceMatrix::checkMul(DevA, DevB, DevC));

  // 16 * 16 = 256 blocks per Thread Block
  dim3 ThrBlockDim{BlockSize, BlockSize};
  // matrix may be bigger than BlockSize, so
  dim3 BlockGridDim{ceilDiv(B.Width, ThrBlockDim.x),
                    ceilDiv(A.Height, ThrBlockDim.y)};
  DEBUG_EXPR(std::cout << "Block grid: {" << BlockGridDim.x << ", "
                       << BlockGridDim.y << "}" << std::endl);
  auto Start = std::chrono::steady_clock::now();
  simpleMatMulKernel<<<BlockGridDim, ThrBlockDim>>>(DevA, DevB, DevC);
  DEBUG_EXPR(cudaDeviceSynchronize());
  checkKernelsExec();
  auto End = std::chrono::steady_clock::now();
  auto Duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(End - Start);
  DEBUG_EXPR(std::cout << "\tKernel duration: " << Duration.count() << "ms"
                       << std::endl);

  auto Res = DeviceMatrix::getHostMat(DevC);
  DevA.free();
  DevB.free();
  DevC.free();
  return Res;
}

__device__ void fillTiles(size_t Iteration, Tile &ATile, DeviceMatrix A,
                          Tile &BTile, DeviceMatrix B) {
  assert(ATile.Size == BTile.Size);
  auto Size = ATile.Size;
  auto CurTilePos = Iteration * Size; // A.X == B.Y
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

__global__ void tiledMatMulKernel(DeviceMatrix A, DeviceMatrix B,
                                  DeviceMatrix C) {
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

  auto Res = 0.0;
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

HostMatrix tiledMatMul(const HostMatrix &A, const HostMatrix &B) {
  DeviceMatrix DevA{A};
  DeviceMatrix DevB{B};
  DeviceMatrix DevC{A.Height, B.Width};
  assert(DeviceMatrix::checkMul(DevA, DevB, DevC));

  // tile size equals BlockSize
  dim3 ThrBlockDim{BlockSize, BlockSize};
  dim3 BlockGridDim{ceilDiv(B.Width, ThrBlockDim.x),
                    ceilDiv(A.Height, ThrBlockDim.y)};

  auto Start = std::chrono::steady_clock::now();
  tiledMatMulKernel<<<BlockGridDim, ThrBlockDim>>>(DevA, DevB, DevC);
  DEBUG_EXPR(cudaDeviceSynchronize());
  checkKernelsExec();
  auto End = std::chrono::steady_clock::now();
  auto Duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(End - Start);
  DEBUG_EXPR(std::cout << "\tKernel duration: " << Duration.count() << "ms"
                       << std::endl);

  DEBUG_EXPR(std::cout << "Tiled finished" << std::endl);
  auto Res = DeviceMatrix::getHostMat(DevC);
  DevA.free();
  DevB.free();
  DevC.free();
  return Res;
}