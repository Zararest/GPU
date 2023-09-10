#pragma once

#include "Matrix.h"

#include <random>

#define MAX_FLOAT 1000

class Generator {
  size_t Seed;

public:
  Generator(size_t Seed) : Seed{Seed} {}

  Matrix generate(size_t H, size_t W) {
    std::mt19937 Rng(Seed);
    std::uniform_int_distribution<std::mt19937::result_type> Dist(1, MAX_FLOAT);

    auto Size = H * W;
    auto Res = Matrix{H, W, new float(Size)};
    for (size_t i = 0; i < Size; ++i)
      Res.Elements[i] = static_cast<float>(Dist(Rng)) / static_cast<float>(Dist(Rng));
    return Res;
    }
};