#include "Matrix.h"
#include "Utils.h"

// Thread block size
constexpr size_t BlockSize = 16;

__global__ void simpleMatMulKernel(DeviceMatrix A, 
                                   DeviceMatrix B, 
                                   DeviceMatrix C) {
  auto Col = blockIdx.x * blockDim.y + threadIdx.x;
  auto Row = blockIdx.y * blockDim.x + threadIdx.y;

  if (Row >= A.Height || Col >= B.Width)
    return;
  
  float Res = 0;
  for (size_t i = 0; i < A.Width; ++i)
    Res += A[Row][i] * B[i][Col];
  C[Row][Col] = Res;
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
  simpleMatMulKernel<<<BlockGridDim, ThrBlockDim>>>(DevA, DevB, DevC);
  checkKernelsExec();

  auto Res = DeviceMatrix::getHostMat(DevC);
  DevA.free();
  DevB.free();
  DevC.free();
  return Res;
}

__device__ void fillTiles(size_t Iteration, Tile &ATile, 
                          DeviceMatrix A, Tile &BTile,
                          DeviceMatrix B) {
  assert(ATile.Size == BTile.Size);
  auto Size = ATile.Size;
  auto CurTilePos = Iteration * Size;    // A.X == B.Y
  ATile.X = CurTilePos;
  BTile.Y = CurTilePos;
  ATile[threadIdx.y][threadIdx.x] = 0.0;                 // this needs to omit check in tile calc
  BTile[threadIdx.y][threadIdx.x] = 0.0;                          
  if (threadIdx.x + ATile.X < A.Width && 
      threadIdx.y + ATile.Y < A.Height)
    ATile[threadIdx.y][threadIdx.x] = A[ATile.Y + threadIdx.y][ATile.X + threadIdx.x];
  
  if (threadIdx.x + BTile.X < B.Width &&
      threadIdx.y + BTile.Y < B.Height)
    BTile[threadIdx.y][threadIdx.x] = B[BTile.Y + threadIdx.y][BTile.X + threadIdx.x];
}

__global__ void tiledMatMulKernel(DeviceMatrix A, DeviceMatrix B,
                                  DeviceMatrix C) {
  auto TileWidth = blockIdx.x;
  auto NumOfTiles = ceilDiv(A.Width, blockDim.x);
  __shared__ float ASharedMem[TileWidth * TileWidth];
  __shared__ float BSharedMem[TileWidth * TileWidth];
  auto ATile =
      Tile{TileWidth, A.Width, /*X*/ 0u, blockIdx.y * TileWidth , ASharedMem};
  auto BTile =
      Tile{TileWidth, B.Width, blockIdx.x * TileWidth, /*Y*/ 0u, BSharedMem};

  auto Res = 0.0;
  for (size_t i = 0; i < NumOfTiles; ++i) {
    fillTiles(i, ATile, A, BTile, B);
    __syncthreads();
    
    for (size_t i = 0; i < TileWidth; ++i)
      Res += ATile[threadIdx.y][i] * BTile[i][threadIdx.x];
    __syncthreads();
  }
  C[threadIdx.y][threadIdx.x] = Res;
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
  tiledMatMulKernel<<<BlockGridDim, ThrBlockDim>>>(DevA, DevB, DevC);
  checkKernelsExec();
}