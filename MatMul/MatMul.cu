#include "Matrix.h"

// Thread block size
#define BLOCK_SIZE 16

__global__ void simpleMatMulKernel(DeviceMatrix &A, 
                                   DeviceMatrix &B, 
                                   DeviceMatrix &C) {
  
}

void simpleMatMul(const HostMatrix A, const HostMatrix B, HostMatrix C) {
  DeviceMatrix DevA{A};
  DeviceMatrix DevB{B};
  DeviceMatrix DevC{C.Width, C.Height};
  assert(DeviceMatrix::checkMul(A, B, C));

  // for small matrix
  auto BlockSize = std::min(BLOCK_SIZE, A.Height, B.Height);

  // 16 * 16 = 256 blocks per Thread Block
  dim3 ThrBlockDim{BlockSize, BlockSize};
  // matrix may be bigger than BLOCK_SIZE, so 
  dim3 BlockGridDim{B.Width / ThrBlockDim.x, A.Height / ThrBlockDim.y};
  simpleMatMulKernel<<<BlockGridDim, ThrBlockDim>>>(DevA, DevB, DevC);

  cudaDeviceSynchronize();

  C = DeviceMatrix::getHostMat(DevC);
}