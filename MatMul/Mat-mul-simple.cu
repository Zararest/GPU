#include <chrono>
#include <iostream>

#include "Matrix.h"
#include "Utils.h"

__global__ void simpleMatMulKernel(DevMatRowMajor A, DevMatRowMajor B,
                                   DevMatRowMajor C) {
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
  DevMatRowMajor DevA{A};
  DevMatRowMajor DevB{B};
  DevMatRowMajor DevC{A.Height, B.Width};
  assert(DevMatRowMajor::checkMul(DevA, DevB, DevC));

  // 16 * 16 = 256 blocks per Thread Block
  dim3 ThrBlockDim{BlockSize, BlockSize};
  // matrix may be bigger than BlockSize, so
  dim3 BlockGridDim{ceilDiv(B.Width, ThrBlockDim.x),
                    ceilDiv(A.Height, ThrBlockDim.y)};
  std::cout << "Block grid: {" << BlockGridDim.x << ", "
                       << BlockGridDim.y << "}" << std::endl;
  auto Start = std::chrono::steady_clock::now();
  simpleMatMulKernel<<<BlockGridDim, ThrBlockDim>>>(DevA, DevB, DevC);
  cudaDeviceSynchronize();
  checkKernelsExec();
  auto End = std::chrono::steady_clock::now();
  auto Duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(End - Start);
  std::cout << "\tKernel duration: " << Duration.count() << "ms"
                       << std::endl;

  auto Res = DevMatRowMajor::getHostMat(DevC);
  DevA.free();
  DevB.free();
  DevC.free();
  return Res;
}