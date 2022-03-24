#include <stdio.h>
#include "cu_prog.h"
#include "gputimer.h"
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; 
    float* elements;
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

// Get a matrix element
// row = height, col = width
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
// row = height, col = width
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
// row = height, col = width
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    // N/BLOCK_SIZE, M/BLOCK_SIZE, 
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
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
// Matrix Multiplication with Shared Memory
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y; // height
    int col = threadIdx.x; // width

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    // K/BLOCK_SIZE, K is not split and compute in one thread
    for (int k = 0; k < (A.width / BLOCK_SIZE); ++k) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, k);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, k, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

// Matrix Multiplication with Shared Memory
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