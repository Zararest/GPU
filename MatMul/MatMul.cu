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
  DEBUG_EXPR(std::cout << "Block grid: {" << BlockGridDim.x << ", " << BlockGridDim.y << "}" << std::endl);
  simpleMatMulKernel<<<BlockGridDim, ThrBlockDim>>>(DevA, DevB, DevC);
  checkKernelsExec();

  auto Res = DeviceMatrix::getHostMat(DevC);
  DevA.free();
  DevB.free();
  DevC.free();
  return Res;
}

__device__  void fillTiles(size_t Iteration, 
                           Tile &A, size_t AHeigth, 
                           Tile &B, size_t BHeigth) {
  assert(A.Size == B.Size);
  auto Size = A.Size;
  auto ATilePos = Iteration * Size;
  for (size_t i = 0; i < Size; ++i)
    for (size_t j = 0; j < Size; ++j) {
      if ()
    }
       
}

__global__ void tiledMatMulKernel(DeviceMatrix A, 
                                  DeviceMatrix B, 
                                  DeviceMatrix C) {
  auto TileWidth = blockIdx.x;
  auto NumOfTiles = ceilDiv(A.Width, blockDim.x);
  __shared__ float ASharedMem[TileWidth * TileWidth];
  __shared__ float BSharedMem[TileWidth * TileWidth];
  auto ATile = Tile{TileWidth, A.Width, /*TilePosX*/ 0u, /*TilePosY*/ 0u, ASharedMem};
  auto BTile = Tile{TileWidth, B.Width, /*TilePosX*/ 0u, /*TilePosY*/ 0u, BSharedMem};

  for (size_t i = 0; i < NumOfTiles; ++i) {

  }
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