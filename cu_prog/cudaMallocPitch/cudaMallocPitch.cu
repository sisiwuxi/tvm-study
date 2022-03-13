#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cu_prog.h"
#include "gputimer.h"


#define ROW 25
#define COLUMN 25

// reason:misaligned address, start address must be aligned
__global__ void memset_pitch(float* A, size_t pitch)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	float *row_a = (float*)((char*)A + i * pitch);
	row_a[j] = 0;
}

__global__ void memset_linear(float* A, size_t pitch)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	if(i < pitch && j < pitch) {
		A[i*pitch + j] = 0;
	}
}

void kernel_cpu(float *A, int row, int column)
{
	int i = 0;
	int j = 0;
	for(i=0; i<row; ++i) {
		float *row_a = (float*)((char*)A + i*column);
		for(j=0; j<column; ++j) {
			row_a[j] = 0;
		}
    }
}
void kernel_cpu_linear(float *A, int row, int column)
{
	int i = 0;
	int j = 0;
	for(i=0; i<row; ++i) {
		for(j=0; j<column; ++j) {
			A[i*column + j] = 0;
		}
    }
}

int main()
{
	float *h_a,*h_from_d;
	float *d_a;
	h_a = (float*)malloc(ROW*COLUMN*sizeof(float));
	h_from_d = (float*)malloc(ROW*COLUMN*sizeof(float));
	
	GpuTimer timer;
	initialMatrix(h_a,ROW,COLUMN);
	// printMatrix(h_a,ROW,COLUMN);
	
	size_t pitch=16;
	CHECK(cudaMallocPitch((void**)&d_a, &pitch, ROW * sizeof(int), COLUMN));
	CHECK(cudaMemcpy2D(d_a, pitch, h_a, ROW * sizeof(int), ROW * sizeof(int), COLUMN, cudaMemcpyHostToDevice));
	dim3 blocksPerGrid(1,1,1);
	dim3 threadsPerBlock(ROW,COLUMN,1);
	timer.Start();
	memset_pitch<<<blocksPerGrid,threadsPerBlock>>>(d_a, pitch);
	timer.Stop();
	CHECK(cudaMemcpy2D(h_a, ROW * sizeof(int), d_a, pitch, ROW * sizeof(int), COLUMN, cudaMemcpyDeviceToHost));
	printf("memset_pitch[%d,%d] Time elapsed = %g ms\n", ROW, COLUMN, timer.Elapsed());
	kernel_cpu(h_a,ROW,COLUMN);
	checkMatrixResult(h_a,h_from_d,ROW,COLUMN);


	// size_t pitch=COLUMN;
	// CHECK(cudaMalloc((void**)&d_a,ROW*COLUMN*sizeof(float)));
	// CHECK(cudaMemcpy(d_a,h_a,ROW*COLUMN*sizeof(float),cudaMemcpyHostToDevice));
	// dim3 blocksPerGrid(1,1,1);
	// dim3 threadsPerBlock(ROW,COLUMN,1);
	// timer.Start();
	// // memset_pitch<<<blocksPerGrid,threadsPerBlock>>>(d_a,pitch);
	// memset_linear<<<blocksPerGrid,threadsPerBlock>>>(d_a,pitch);
	// timer.Stop();
	// CHECK(cudaMemcpy(h_from_d,d_a,ROW*COLUMN*sizeof(float),cudaMemcpyDeviceToHost));
	// printf("memset_linear[%d,%d] Time elapsed = %g ms\n", ROW, COLUMN, timer.Elapsed());
	// kernel_cpu_linear(h_a,ROW,COLUMN);
	// checkResult(h_a,h_from_d,ROW*COLUMN);

	free(h_a);
	CHECK(cudaFree(d_a));
	return 0;
}