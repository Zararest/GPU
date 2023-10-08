#include <chrono>
#include <iostream>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BlockSize 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BlockSize
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BlockSize, BlockSize);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    std::cout << "Block grid: {" << dimGrid.x << ", "
                       << dimGrid.y << "}" << std::endl;

    auto Start = std::chrono::steady_clock::now();
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    auto End = std::chrono::steady_clock::now();
    auto Duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(End - Start);
    std::cout << "\tKernel duration: " << Duration.count() << "ms"
                        << std::endl;

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

int main() {
    size_t Heigth = 4096;
    size_t Width = 4096;
    size_t JointSize = 4096; 
    Matrix A{Heigth, JointSize, new float[Heigth * JointSize]};
    Matrix B{JointSize, Width, new float[Width * JointSize]};
    Matrix C{Heigth, Width, new float[Width * Heigth]};
   
    auto Start = std::chrono::steady_clock::now();
    MatMul(A, B, C);
    auto End = std::chrono::steady_clock::now();
    auto Duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(End - Start);
    std::cout << "Without shared:" << ": " << Duration.count() << "ms" << std::endl;
}