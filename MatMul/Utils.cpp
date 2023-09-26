#include "Utils.h"
#include "Matrix.h"

#include <random>

#define MAX_FLOAT 10.0

HostMatrix generate(size_t Height, size_t Width) {
  std::mt19937 Rng(4);
  std::uniform_real_distribution<> Dist(1.0, MAX_FLOAT);

  auto Size = Height * Width;
  auto Res = HostMatrix{Height, Width};
  for (auto &It : Res.Elements)
    It = Dist(Rng);
  return Res;
}

HostMatrix referenceMul(HostMatrix &A, HostMatrix &B) {
  assert(A.Width == B.Height);
  auto NewMatrix = HostMatrix{A.Height, B.Width};
  for (size_t Row = 0; Row < A.Height; ++Row)
    for (size_t Col = 0; Col < B.Width; ++Col)
      for (size_t k = 0; k < A.Width; ++k)
        NewMatrix.get(Row, Col) += A.get(Row, k) * B.get(k, Col);
  return NewMatrix;
}

void printDeviceLimits(std::ostream &S) {
  int DeviceCount;
  cudaGetDeviceCount(&DeviceCount);
  S << "Number of the devices: " << DeviceCount << "\n" << std::endl;

  for (int i = 0; i < DeviceCount; ++i) {
    struct cudaDeviceProp Props;
    cudaGetDeviceProperties(&Props, i);
    S << "Parameters of the device " << i << ":" << std::endl;
    S << "\tGlobal memory size: " << Props.totalGlobalMem << "\n"
      << "\tShared memory size (per block): " << Props.sharedMemPerBlock << "\n"
      << "\tConstant memory size: " << Props.totalConstMem << "\n"
      << "\tRegs per block: " << Props.regsPerBlock << "\n"
      << "\tMax threads per block: " << Props.maxThreadsPerBlock << "\n"
      << "\tMax threads dim: {" << Props.maxThreadsDim[0] << ", "
      << Props.maxThreadsDim[1] << ", " << Props.maxThreadsDim[2] << "}\n"
      << "\tMax grid dim: {" << Props.maxGridSize[0] << ", "
      << Props.maxGridSize[1] << ", " << Props.maxGridSize[2] << "}\n"
      << "\tClock rate: " << Props.clockRate << std::endl;
  }
}

void checkKernelsExec() {
  auto Err = cudaGetLastError();
  if (Err != cudaSuccess) {
    std::cout << "Kernel terminated with error: " << cudaGetErrorString(Err)
              << std::endl;
    exit(1);
  }
}