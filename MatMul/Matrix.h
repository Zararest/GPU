#pragma once

#include "Utils.h"

#include <algorithm>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

// Thread block size
constexpr size_t BlockSize = 16;

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

struct HostMatrixInt final {
  size_t Width = 0;
  size_t Height = 0;
  std::vector<int> Elements;

  HostMatrixInt(size_t Height, size_t Width)
      : Width{Width}, Height{Height}, Elements(Height * Width) {}

  inline int &get(size_t RowNum, size_t ColNum) {
    DEBUG_EXPR(assert(RowNum < Height));
    DEBUG_EXPR(assert(ColNum < Width));
    return Elements[RowNum * Width + ColNum];
  }
};

struct DevMatRowMajor final {
  size_t Width = 0;
  size_t Height = 0;
  float *Elements;

  __host__ DevMatRowMajor(const HostMatrix &HMat)
      : Width{HMat.Width}, Height{HMat.Height} {
    auto Size = Width * Height * sizeof(float);
    CUDA_CHECK(cudaMalloc((void **)&Elements, Size));
    CUDA_CHECK(cudaMemcpy(Elements, HMat.Elements.data(), Size,
                          cudaMemcpyHostToDevice));
  }

  __host__ DevMatRowMajor(size_t Height, size_t Width)
      : Width{Width}, Height{Height} {
    auto Size = Width * Height * sizeof(float);
    cudaMalloc((void **)&Elements, Size);
  }

  __host__ void free() { CUDA_CHECK(cudaFree(Elements)); }

  __host__ __device__ static bool checkMul(const DevMatRowMajor &A,
                                           const DevMatRowMajor &B,
                                           const DevMatRowMajor &C) {
    return A.Width == B.Height && C.Height == A.Height && C.Width == B.Width;
  }

  __host__ static HostMatrix getHostMat(const DevMatRowMajor &DevMat) {
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

struct DevMatRowMajorInt final {
  size_t Width = 0;
  size_t Height = 0;
  int *Elements;

  __host__ DevMatRowMajorInt(const HostMatrixInt &HMat)
      : Width{HMat.Width}, Height{HMat.Height} {
    auto Size = Width * Height * sizeof(int);
    CUDA_CHECK(cudaMalloc((void **)&Elements, Size));
    CUDA_CHECK(cudaMemcpy(Elements, HMat.Elements.data(), Size,
                          cudaMemcpyHostToDevice));
  }

  __host__ DevMatRowMajorInt(size_t Height, size_t Width)
      : Width{Width}, Height{Height} {
    auto Size = Width * Height * sizeof(int);
    cudaMalloc((void **)&Elements, Size);
  }

  __host__ void free() { CUDA_CHECK(cudaFree(Elements)); }

  __host__ __device__ static bool checkMul(const DevMatRowMajorInt &A,
                                           const DevMatRowMajorInt &B,
                                           const DevMatRowMajorInt &C) {
    return A.Width == B.Height && C.Height == A.Height && C.Width == B.Width;
  }

  __host__ static HostMatrixInt getHostMat(const DevMatRowMajorInt &DevMat) {
    HostMatrixInt HostMat{DevMat.Height, DevMat.Width};
    auto SizeInFloat = DevMat.Width * DevMat.Height;
    auto *Buf = new int[SizeInFloat];
    CUDA_CHECK(cudaMemcpy(Buf, DevMat.Elements, SizeInFloat * sizeof(int),
                          cudaMemcpyDeviceToHost));
    HostMat.Elements.clear();
    std::copy(Buf, Buf + SizeInFloat, std::back_inserter(HostMat.Elements));
    delete[] Buf;
    return HostMat;
  }

  __device__ inline int &get(size_t RowNum, size_t ColNum) {
    DEBUG_EXPR(assert(RowNum < Height));
    DEBUG_EXPR(assert(ColNum < Width));
    return Elements[RowNum * Width + ColNum];
  }
};

class DevMatColMajor final {
  __host__ static void transposeHost(HostMatrix &Mat) {
    for (size_t X = 0; X < Mat.Width; ++X)
      for (size_t Y = X; Y < Mat.Height; ++Y)
        std::swap(Mat.get(X, Y), Mat.get(Y, X));
  }

public:
  size_t Width = 0;
  size_t Height = 0;
  float *Elements;

  __host__ DevMatColMajor(HostMatrix HMat)
      : Width{HMat.Width}, Height{HMat.Height} {
    transposeHost(HMat);
    auto Size = Width * Height * sizeof(float);
    CUDA_CHECK(cudaMalloc((void **)&Elements, Size));
    CUDA_CHECK(cudaMemcpy(Elements, HMat.Elements.data(), Size,
                          cudaMemcpyHostToDevice));
  }

  __host__ void free() { CUDA_CHECK(cudaFree(Elements)); }

  __device__ inline float &get(size_t RowNum, size_t ColNum) {
    DEBUG_EXPR(assert(RowNum < Height));
    DEBUG_EXPR(assert(ColNum < Width));
    return Elements[ColNum * Height + RowNum];
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

struct TileInt {
  size_t Size = 0; // size of the square tile
  size_t X = 0;    // column of the tile
  size_t Y = 0;    // row of the tile
  int *Elements; // elements of the real matrix

  __device__ TileInt(size_t Size, size_t X, size_t Y, int *Elements)
      : Size{Size}, X{X}, Y{Y}, Elements{Elements} {}

  __device__ inline int &get(size_t RowNum, size_t ColNum) {
    DEBUG_EXPR(assert(RowNum < Size));
    DEBUG_EXPR(assert(ColNum < Size));
    return Elements[RowNum * Size + ColNum];
  }
};

HostMatrix transposeTiledMatMul(const HostMatrix &A, const HostMatrix &B, bool PrintTime);
HostMatrix tiledMatMul(const HostMatrix &A, const HostMatrix &B, bool PrintTime);
HostMatrixInt tiledMatMulInt(const HostMatrixInt &A, const HostMatrixInt &B, bool PrintTime);
HostMatrix simpleMatMul(const HostMatrix &A, const HostMatrix &B, bool PrintTime);