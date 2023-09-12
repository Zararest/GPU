#pragma once

#include "Utils.h"

#include <algorithm>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

class HostMatrix final {
  struct Proxy {
    std::vector<float>::iterator It;
    size_t Width = 0;

    Proxy(std::vector<float>::iterator It, size_t Width)
        : It{It}, Width{Width} {}

    float &operator[](size_t ColNum) {
      assert(ColNum < Width);
      return *(It + ColNum);
    }
  };

public:
  size_t Width = 0;
  size_t Height = 0;
  std::vector<float> Elements;

  HostMatrix(size_t Height, size_t Width)
      : Width{Width}, Height{Height}, Elements(Height * Width) {}

  Proxy operator[](size_t RowNum) {
    assert(RowNum < Height);
    return Proxy{Elements.begin() + RowNum * Width, Width};
  }

  void print(std::ostream &S) {
    for (size_t i = 0; i < Height; ++i) {
      for (size_t j = 0; j < Width; ++j)
        std::cout << (*this)[i][j] << " ";
      std::cout << std::endl;
    }
  }
};

class DeviceMatrix final {
  struct Proxy {
    float *RowPtr;
    size_t Width = 0;

    __device__ Proxy(float *RowPtr, size_t Width)
        : RowPtr{RowPtr}, Width{Width} {}

    __device__ float &operator[](size_t ColNum) {
      assert(ColNum < Width);
      return RowPtr[ColNum];
    }
  };

public:
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

  __device__ Proxy operator[](size_t RowNum) {
    assert(RowNum < Height);
    return {Elements + RowNum * Width, Width};
  }
};

class Tile {
  struct Proxy {
    size_t TileSize = 0;
    float *RowPtr;

    float &operator[](size_t ColNum) {
      assert(ColNum < TileSize);
      return RowPtr[ColNum];
    }
  };

public:
  size_t Size = 0;      // size of the square tile
  size_t MatrWidth = 0; // width of the real matrix
  size_t X = 0;         // column of the tile
  size_t Y = 0;         // row of the tile
  float *Elements;      // elements of the real matrix

  __device__ Proxy operator[](size_t RowNum) {
    assert(RowNum < Size);
    auto RowPtr = Elements + RowNum * MatrWidth + TilePos;
    return Proxy{Size, RowPtr};
  }
};

void sharedMatMul(const HostMatrix A, const HostMatrix B, HostMatrix C);
HostMatrix simpleMatMul(const HostMatrix &A, const HostMatrix &B);