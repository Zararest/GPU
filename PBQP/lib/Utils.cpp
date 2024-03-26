#include "Utils.h"
#include "Matrix.h"

#include <random>

namespace utils {

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

void reportFatalError(const std::string &Msg) {
  std::cerr << "Error: " << Msg << std::endl;
  exit(-1);
}

bool isEqual(double Lhs, double Rhs) {
  constexpr auto e = 0.01;
  auto Delta = Lhs * e;
  return Rhs >= Lhs - Delta && Rhs <= Lhs + Delta;
}

} // namesapce utils