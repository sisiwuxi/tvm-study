#include <stdio.h>
#include "cu_prog.h"
#include "gputimer.h"
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct
{
    int width;
    int height;
    float *elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE (16)
#define MATMUL_M (8*BLOCK_SIZE)
#define MATMUL_N (64*BLOCK_SIZE)
#define MATMUL_K (128*BLOCK_SIZE)



void matmulCPU(const Matrix A, const Matrix B, Matrix C)
{
    // A[M,K]
    // B[K,N]
    // C[M,N]
	int M = A.height;
	int N = B.width;
    int K = A.width;//B.height
    int m,n,k;
    float value = 0;
	for(m=0; m<M; ++m) {
		for(n=0; n<N; ++n) {
            value = 0;
            for(k=0; k<K; ++k) {
                value += A.elements[m*K + k] * B.elements[k*N + n];
            }
            C.elements[m*N + n] = value;
		}
    }
}

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(C.width / dimBlock.x, C.height / dimBlock.y);
    printf("\ngrid(%d,%d,%d), block(%d,%d,%d)\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
// Matrix Multiplication without Shared Memory
// __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
// {
//     // Each thread computes one element of C
//     // by accumulating results into Cvalue
//     float Cvalue = 0;
//     int row = blockIdx.x * blockDim.x + threadIdx.x;//m
//     int col = blockIdx.y * blockDim.y + threadIdx.y;//n
//     for (int k = 0; k < A.height; ++k)//k
//         Cvalue += A.elements[row*A.height + k] * B.elements[k*B.height + col];
//     C.elements[row * C.height + col] = Cvalue;
// }
// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;//m,height
    int col = blockIdx.x * blockDim.x + threadIdx.x;//n,width
    for (int k = 0; k < A.width; ++k)
        Cvalue += A.elements[row * A.width + k]
                * B.elements[k * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

// Matrix Multiplication without Shared Memory
int main()
{
    GpuTimer timer;
    Matrix A, B, C, D;
    A.width = MATMUL_K;
    A.height = MATMUL_M;
    A.elements = (float *)malloc(A.width * A.height * sizeof(float));
    B.width = MATMUL_N;
    B.height = MATMUL_K;
    B.elements = (float *)malloc(B.width * B.height * sizeof(float));
    C.width = MATMUL_N;
    C.height = MATMUL_M;
    C.elements = (float *)malloc(C.width * C.height * sizeof(float));
    initialMatrix(A.elements, A.width, A.height);
    initialMatrix(B.elements, B.width, B.height);
    timer.Start();
    MatMul(A, B, C);
	timer.Stop();
    printf("Matrix Multiplication without Shared Memory[%d,%d] with split[%d,%d] Time elapsed = %g ms\n", A.width, A.height, B.width, B.height, timer.Elapsed());
    D.width = MATMUL_N;
    D.height = MATMUL_M;
    D.elements = (float *)malloc(D.width * D.height * sizeof(float));
    matmulCPU(A, B, D);
    if (checkMatrixResult(D.elements,C.elements,C.width,C.height) == false) {
        printMatrix(A.elements, A.width, A.height);
        printMatrix(B.elements, B.width, B.height);
        printMatrix(C.elements, C.width, C.height);
    }
    
    free(A.elements);
    free(B.elements);
    free(C.elements);
    free(D.elements);
}