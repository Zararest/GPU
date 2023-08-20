#include <iostream>

__global__ void Parameters() {
  printf("Сетка блоков: %i, %i, %i\n", gridDim.x, gridDim.y, gridDim.z);
  printf("Треды в блоках: %i, %i, %i\n", blockDim.x, blockDim.y, blockDim.z);
}

int main() {
  Parameters<<<1, 1>>>();
  cudaDeviceSynchronize();
}