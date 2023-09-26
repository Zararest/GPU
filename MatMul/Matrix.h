#pragma once

#include "Utils.h"

#include <algorithm>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

struct HostMatrix final {
  size_t Width = 0;
  size_t Height = 0;
  std::vector<float> Elements;

  HostMatrix(size_t Height, size_t Width)
      : Width{Width}, Height{Height}, Elements(Height * Width) {}

  inline float &get(size_t RowNum, size_t ColNum) {
    DEBUG_EXPR(assert(RowNum < Height));
    DEBUG_EXPR(assert(ColNum < Width));
    return Elements[RowNum * Width + ColNum];
  }

  void print(std::ostream &S) {
    for (size_t i = 0; i < Height; ++i) {
      for (size_t j = 0; j < Width; ++j)
        std::cout << get(i, j) << " ";
      std::cout << std::endl;
    }
  }
};

struct DeviceMatrix final {
  size_t Width = 0;
  size_t Height = 0;
  float *Elements;

  __host__ DeviceMatrix(const HostMatrix &HMat)
      : Width{HMat.Width}, Height{HMat.Height} {
    auto Size = Width * Height * sizeof(float);
    CUDA_CHECK(cudaMalloc((void **)&Elements, Size));
    CUDA_CHECK(cudaMemcpy(Elements, HMat.Elements.data(), Size,
                          cudaMemcpyHostToDevice));
  }

  __host__ DeviceMatrix(size_t Height, size_t Width)
      : Width{Width}, Height{Height} {
    auto Size = Width * Height * sizeof(float);
    cudaMalloc((void **)&Elements, Size);
  }

  __host__ void free() { CUDA_CHECK(cudaFree(Elements)); }

  __host__ __device__ static bool checkMul(const DeviceMatrix &A,
                                           const DeviceMatrix &B,
                                           const DeviceMatrix &C) {
    return A.Width == B.Height && C.Height == A.Height && C.Width == B.Width;
  }

  __host__ static HostMatrix getHostMat(const DeviceMatrix &DevMat) {
    HostMatrix HostMat{DevMat.Height, DevMat.Width};
    auto SizeInFloat = DevMat.Width * DevMat.Height;
    auto *Buf = new float[SizeInFloat];
    CUDA_CHECK(cudaMemcpy(Buf, DevMat.Elements, SizeInFloat * sizeof(float),
                          cudaMemcpyDeviceToHost));
    HostMat.Elements.clear();
    std::copy(Buf, Buf + SizeInFloat, std::back_inserter(HostMat.Elements));
    delete[] Buf;
    return HostMat;
  }

  __device__ inline float &get(size_t RowNum, size_t ColNum) {
    DEBUG_EXPR(assert(RowNum < Height));
    DEBUG_EXPR(assert(ColNum < Width));
    return Elements[RowNum * Width + ColNum];
  }
};

struct Tile {
  size_t Size = 0; // size of the square tile
  size_t X = 0;    // column of the tile
  size_t Y = 0;    // row of the tile
  float *Elements; // elements of the real matrix

  __device__ Tile(size_t Size, size_t X, size_t Y, float *Elements)
      : Size{Size}, X{X}, Y{Y}, Elements{Elements} {}

  __device__ inline float &get(size_t RowNum, size_t ColNum) {
    DEBUG_EXPR(assert(RowNum < Size));
    DEBUG_EXPR(assert(ColNum < Size));
    return Elements[RowNum * Size + ColNum];
  }
};

HostMatrix tiledMatMul(const HostMatrix &A, const HostMatrix &B);
HostMatrix simpleMatMul(const HostMatrix &A, const HostMatrix &B);