#pragma once

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <cassert>

class Matrix {
  struct Proxy {
    float *RowPtr = nullptr;
    size_t Width = 0;

    __host__ __device__
    Proxy(float *RowPtr, size_t Width) : RowPtr{RowPtr}, Width{Width} {}

    __host__ __device__
    float &operator [](size_t ColNum) {
      assert(RowPtr);
      assert(ColNum < Width);
      return RowPtr[ColNum];
    }
  };

public:
  size_t Width = 0;
  size_t Height = 0;
  float *Elements = nullptr;

  __host__ __device__
  Proxy operator [](size_t RowNum) {
    return Proxy{Elements + RowNum, Width};
  }

  __host__
  void print(std::ostream &S) {
    for (size_t i = 0; i < Height; ++i) {
      for (size_t j = 0; j < Width; ++j)
        std::cout << (*this)[i][j] << " ";
    std::cout << std::endl;
    }
  }
};