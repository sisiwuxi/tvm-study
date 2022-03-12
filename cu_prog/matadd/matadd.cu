#include <stdio.h>
#include "cuda_profiler_api.h"
#include "cu_prog.h"
#include "gputimer.h"

#define ROW 64
#define COLUMN 64
#define SR 32
#define SC 32
#define MAX_NUM_THRE_PER_BLOCK 1024

// Kernel invocation with one block of ROW * COLUMN * 1 threads
__global__ void MatAdd(float *A, float *B, float *C, int row, int column)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // C = A + B (write down your codes)
    if(i < row && j < column)
    {
		C[i*column + j] = A[i*column + j] + B[i*column + j];
    }

}

void matAddCPU(float *A, float *B, float *C, int row, int column)
{
	int i = 0;
	int j = 0;
	for(i=0; i<row; ++i) {
		for(j=0; j<column; ++j) {
			C[i*column + j] = A[i*column + j] + B[i*column + j];
		}
    }
}

int main(void)
{
	GpuTimer timer;
	float *h_a,*h_b,*h_c,*h_from_d;
	float *d_a,*d_b,*d_c; //device variables 
	
	h_a = (float*)malloc(ROW*COLUMN*sizeof(float));
	h_b = (float*)malloc(ROW*COLUMN*sizeof(float));
	h_c = (float*)malloc(ROW*COLUMN*sizeof(float));
    h_from_d = (float*)malloc(ROW*COLUMN*sizeof(float));

	//cuda memory allocation on the device
	CHECK(cudaMalloc((void**)&d_a,ROW*COLUMN*sizeof(float)));
	CHECK(cudaMalloc((void**)&d_b,ROW*COLUMN*sizeof(float)));
	CHECK(cudaMalloc((void**)&d_c,ROW*COLUMN*sizeof(float)));

	printf("initialData to all array...\n");
    initialMatrix(h_a,ROW,COLUMN);
    initialMatrix(h_b,ROW,COLUMN);
    initialMatrix(h_c,ROW,COLUMN);

	//cuda memory copy from host to device
	CHECK(cudaMemcpy(d_a,h_a,ROW*COLUMN*sizeof(float),cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_b,h_b,ROW*COLUMN*sizeof(float),cudaMemcpyHostToDevice));

	printf("launch vector adding kernel...\n");
	CHECK(cudaProfilerStart());
	dim3 blocksPerGrid((ROW+SR-1)/SR,(COLUMN+SC-1)/SC,1);
	dim3 threadsPerBlock(SR,SC,1);
	timer.Start();
	MatAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a,d_b,d_c,ROW,COLUMN);
	timer.Stop();
	CHECK(cudaProfilerStop());

	//CPU equivalent
	matAddCPU(h_a,h_b,h_from_d,ROW,COLUMN);
	// //cuda memory copy from device to host
	CHECK(cudaMemcpy(h_c,d_c,ROW*COLUMN*sizeof(float),cudaMemcpyDeviceToHost));

    checkResult(h_from_d,h_c,ROW*COLUMN);

	free(h_a);
	free(h_b);
	free(h_c);
    free(h_from_d);

	CHECK(cudaFree(d_a));
	CHECK(cudaFree(d_b));
	CHECK(cudaFree(d_c));

	printf("matadd[%d,%d] with split[%d,%d] Time elapsed = %g ms\n", ROW, COLUMN, SR, SC, timer.Elapsed());
	return 0;

}