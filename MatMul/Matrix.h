#pragma once

#include <vector>
#include <algorithm>

struct HostMatrix final {
  int Width = 0;
  int Height = 0;
  // M(row, col) = *(M.elements + row * M.stride + col)
  std::vector<float> Elements; 
};

class DeviceMatrix final {
  int Width = 0;
  int Height = 0;
  float* Elements = nullptr;

public:
  DeviceMatrix(const HostMatrix &HMat) : Width{HMat.Width},
                                         Height{HMat.Height} {
    auto Size = Width * Height * sizeof(float);
    cudaMalloc(&Elements, Size);
    cudaMemcpy(Elements, HMat.Elements.data(), Size, 
               cudaMemcpyHostToDevice);
  }

  DeviceMatrix(int Width, int Height) : Width{Width}, Height{Height} {
    auto Size = Width * Height * sizeof(float);
    cudaMalloc(&Elements, Size);
  }

  ~DeviceMatrix() {
    cudaFree(Elements);
  }

  static bool checkMul(const DeviceMatrix &A, const DeviceMatrix &B,
                       const DeviceMatrix &C) {
    return A.Width == B.Height && C.Height == A.Height && C.Width == B.Width;
  }

  static HostMatrix getHostMat(const DeviceMatrix &DevMat) {
    HostMatrix HostMat{DevMat.Width, DevMat.Height};
    auto SizeInFloat = DevMat.Width * DevMat.Height;
    auto *Buf = new float(SizeInFloat);
    cudaMemcpy(Buf, DevMat.Elements, SizeInFloat + sizeof(float), 
               cudaMemcpyDeviceToHost);
    std::copy(Buf, Buf + SizeInFloat, std::back_inserter(HostMat.Elements));
    return HostMat;
  }
};

class MatrixView final {
  int Width = 0;
  int Height = 0;
  int Stride = 0;
  float* elements = nullptr;
};

void sharedMatMul(const HostMatrix A, const HostMatrix B, HostMatrix C);
void simpleMatMul(const HostMatrix A, const HostMatrix B, HostMatrix C);