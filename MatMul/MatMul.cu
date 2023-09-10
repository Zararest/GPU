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

  // for small matrix
  auto RealBlockSize = std::min(BlockSize, std::min(A.Height, B.Height));

  // 16 * 16 = 256 blocks per Thread Block
  dim3 ThrBlockDim{RealBlockSize, RealBlockSize};
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