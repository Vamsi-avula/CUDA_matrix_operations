# CUDA_matrix_operations
I am developing a CUDA-based GPU program to perform various matrix operations. This implementation will incorporate advanced CUDA concepts such as memory coalescing, efficient memory allocation strategies, and other performance-oriented optimizations.
CUDA Matrix Operations – Addition & Multiplication

This project demonstrates GPU-accelerated matrix operations using NVIDIA CUDA.
It includes two fundamental linear-algebra computations:

Matrix Addition

Matrix Multiplication

Both are implemented using CUDA kernels with configurable grid/block dimensions.
The project is designed as a clear introductory example of GPU parallelism, memory management, and computational acceleration using CUDA.

⭐ Features

CUDA kernel for matrix addition

CUDA kernel for matrix multiplication

GPU memory allocation using cudaMalloc

Host-to-device and device-to-host transfers using cudaMemcpy

2D thread indexing for accessing matrix elements

Parameterized matrix sizes (m, k, n)

Grid/block configuration using dim3

Example initialization and formatted output

Supports compute capability 7.5 (sm_75)

🧠 CUDA Concepts Demonstrated

This project touches on several important CUDA concepts:

✔ Thread hierarchy

2D blocks and 2D grids for matrix mapping

Mapping (row, col) to global thread index

✔ Memory management

Global GPU memory allocation

Data transfers between CPU and GPU

Explicit memory cleanup

✔ Performance-oriented practices (intro level)

Thread-safe boundary checks

Coalesced memory access patterns

Synchronization using cudaDeviceSynchronize()

📂 File Structure
├── matrix_cuda.cu     # Main CUDA implementation (kernels + host code)
└── README.md          # Project documentation

🚀 Build and Run Instructions
Prerequisites

NVIDIA GPU with CUDA support

CUDA Toolkit installed

NVCC compiler

Compile
nvcc matrix_cuda.cu -o matrix_cuda

Run
./matrix_cuda

🧩 Code Overview
Matrix Addition Kernel
__global__ void matAdd(float *A, float *B, float *C, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
        C[row * n + col] = A[row * n + col] + B[row * n + col];
}

Matrix Multiplication Kernel
__global__ void matMul(float *A, float *B, float *C, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        float sum = 0.0f;
        for (int i = 0; i < k; i++)
            sum += A[row * k + i] * B[i * n + col];

        C[row * n + col] = sum;
    }
}

Launch Configuration
dim3 block(16, 16);
dim3 grid((n + 15) / 16, (m + 15) / 16);

🖥 Example Output

The program prints:

Matrix A

Matrix B

Result of A + B

Result of A × B

Example snippet:

Matrix A:
 0.0  1.0  2.0
 1.0  2.0  3.0
 2.0  3.0  4.0

Matrix B:
 0.0  1.0  2.0
 2.0  3.0  4.0
 4.0  5.0  6.0

Matrix Addition (A + B):
 ...

Matrix Multiplication (A × B):
 ...

📈 Possible Enhancements (Future Work)

You can extend this project by implementing:

Shared memory tiling for optimized matrix multiplication

Loop unrolling for performance

Memory coalescing improvements

Benchmarking (GPU time, GFLOPS)

Support for non-square matrices

Error checking using CUDA error macros

Unified Memory (cudaMallocManaged) version

If you want, I can generate these optimized versions too.

📜 License

MIT License (or your preferred license).
