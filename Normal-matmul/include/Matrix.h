#pragma once

#include "Utils.h"

#include <algorithm>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <chrono>
#include <iostream>

namespace host {

template <typename T>
class Matrix {
  using It = typename std::vector<T>::iterator;

  size_t Width = 0;
  size_t Height = 0;
  std::vector<T> Elements;

  class Proxy {
    std::vector<T> &Elems;
    size_t Width = 0;
    size_t RowNum = 0;

  public:
    __host__
    Proxy(size_t RowNum, size_t Width, std::vector<T> &Elems) 
        : Elems{Elems}, Width{Width}, RowNum{RowNum} {}

    __host__
    T &operator [](size_t Col) {
      DEBUG_EXPR(assert(Col < Width));
      return Elems[RowNum * Width + Col];
    }
  };

  class ConstProxy {
    const std::vector<T> &Elems;
    size_t Width = 0;
    size_t RowNum = 0;

  public:
    __host__
    ConstProxy(size_t RowNum, size_t Width, const std::vector<T> &Elems) 
        : Elems{Elems}, Width{Width}, RowNum{RowNum} {}

    __host__
    const T &operator [](size_t Col) const {
      DEBUG_EXPR(assert(Col < Width));
      return Elems[RowNum * Width + Col];
    }
  };

public:
  __host__
  Matrix(size_t Height = 0, size_t Width = 0)
      : Width{Width}, Height{Height}, Elements(Height * Width) {}

  template <typename It>
  __host__
  Matrix(It Beg, It End, size_t Height, size_t Width) 
      : Width{Width}, Height{Height}, Elements{Beg, End} {}

  __host__
  Proxy operator [](size_t Row) {
    DEBUG_EXPR(assert(Row < Height));
    return Proxy{Row, Width, Elements};
  }

  __host__
  ConstProxy operator [](size_t Row) const {
    DEBUG_EXPR(assert(Row < Height));
    return ConstProxy{Row, Width, Elements};
  }

   __host__
  const T *data() const {
    return Elements.data();
  }

   __host__
  It begin() {
    return Elements.begin();
  }

   __host__
  It end() {
    return Elements.end();
  }

   __host__
  size_t w() const {
    return Width;
  }

   __host__
  size_t h() const {
    return Height;
  }

  __host__
  static Matrix<T> transpose(const Matrix<T> &A) {
    auto Res = Matrix<T>{A.w(), A.h()};
    for (size_t i = 0; i < A.h(); ++i)
      for (size_t j = 0; j < A.w(); ++j)
        Res[j][i] = A[i][j];
    return Res; 
  }
};

template <typename T>
struct MatMulResult {
  Matrix<T> Matr;
  long DurationMs;
};

} // namespace host

namespace device {

template <typename T>
class Matrix {
  size_t Width = 0;
  size_t Height = 0;
  T *Elements;

  class Proxy {
    T *RowElems;
    size_t Width = 0;

  public:
    __device__
    Proxy(size_t Width, T *RowElems) : RowElems{RowElems}, Width{Width} {}

    __device__
    T &operator [](size_t Col) {
      DEBUG_EXPR(assert(Col < Width));
      return RowElems[Col];
    }
  };

public:
  __host__
  Matrix(size_t Height, size_t Width) : Width{Width}, Height{Height} {
    auto Size = Width * Height * sizeof(float);
    CUDA_CHECK(cudaMalloc((void **)&Elements, Size));
  }

  __host__
  Matrix(const host::Matrix<T> &HostMat) : Width{HostMat.w()}, Height{HostMat.h()} {
    auto Size = Width * Height * sizeof(T);
    CUDA_CHECK(cudaMalloc((void **)&Elements, Size));
    CUDA_CHECK(cudaMemcpy(Elements, HostMat.data(), Size,
                          cudaMemcpyHostToDevice));
  }

  __host__ 
  void free() { CUDA_CHECK(cudaFree(Elements)); }

  __host__
  host::Matrix<T> getHostMatrix() const {
    auto SizeInType = Width * Height;
    auto *Buf = new T[SizeInType];
    CUDA_CHECK(cudaMemcpy(Buf, Elements, SizeInType * sizeof(T),
                          cudaMemcpyDeviceToHost));
    auto HostMat = host::Matrix<T>{Buf, Buf + SizeInType, 
                                Height, Width};
    delete[] Buf;
    return HostMat;
  }

  __device__
  Proxy operator[](size_t Row) {
    DEBUG_EXPR(assert(Row < Height));
    return Proxy{Width, Elements + Row * Width};
  }

  __device__
  size_t w() const {
    return Width;
  }

  __device__
  size_t h() const {
    return Height;
  }

  class Tile {
    size_t Size = 0;
    T *Elements;

  public:
    __device__
    Tile(size_t Size, T *Elements) : Size{Size}, Elements{Elements} {}

    __device__
    Proxy operator[](size_t Row) {
      DEBUG_EXPR(assert(Row < Size));
      return Proxy{Size, Elements + Row * Size};
    }
  };
};
} // namespace device
