#include <stdio.h>
#include "cuda_profiler_api.h"
#include "cu_prog.h"
#include "gputimer.h"

#define SIZE 1025
#define MAX_NUM_THRE_PER_BLOCK 1024

// Kernel invocation with N threads
//    Maximum number of threads per block:1024
//    only success when SIZE <= 1024, use 1 grid with 1 block
__global__ void vecAdd_1(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

// Kernel invocation with blocks x threads
__global__ void vecAdd_2(float* A, float* B, float* C, int len)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < len) {
        C[i] = A[i] + B[i];
    }
}

void vecAddCPU(float* A, float* B, float* C, int len)
{
	int i = 0;
	for(i=0; i<len; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main(void)
{
    GpuTimer timer;
	float *h_a,*h_b,*h_c,*h_from_d;
	float *d_a,*d_b,*c_dev; //device variables 
	
	h_a = (float*)malloc(SIZE*sizeof(float));
	h_b = (float*)malloc(SIZE*sizeof(float));
	h_c = (float*)malloc(SIZE*sizeof(float));
    h_from_d = (float*)malloc(SIZE*sizeof(float));

	//cuda memory allocation on the device
	CHECK(cudaMalloc((void**)&d_a,SIZE*sizeof(float)));
	CHECK(cudaMalloc((void**)&d_b,SIZE*sizeof(float)));
	CHECK(cudaMalloc((void**)&c_dev,SIZE*sizeof(float)));

	printf("initialData to all array...\n");
    initialData(h_a,SIZE);
    initialData(h_b,SIZE);
    initialData(h_c,SIZE);

	//cuda memory copy from host to device
	CHECK(cudaMemcpy(d_a,h_a,SIZE*sizeof(float),cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_b,h_b,SIZE*sizeof(float),cudaMemcpyHostToDevice));

	printf("launch vector adding kernel...\n");
	CHECK(cudaProfilerStart());
	// vecAdd_1<<<1, SIZE>>>(d_a,d_b,c_dev);
	dim3 blocksPerGrid((SIZE+(MAX_NUM_THRE_PER_BLOCK-1))/MAX_NUM_THRE_PER_BLOCK,1,1);
	dim3 threadsPerBlock(MAX_NUM_THRE_PER_BLOCK,1,1);
    timer.Start(); 
    vecAdd_2<<<blocksPerGrid, threadsPerBlock>>>(d_a,d_b,c_dev,SIZE);
    timer.Stop();
	CHECK(cudaProfilerStop());

	//CPU equivalent
	vecAddCPU(h_a,h_b,h_from_d,SIZE);
	//cuda memory copy from device to host
	CHECK(cudaMemcpy(h_c,c_dev,SIZE*sizeof(float),cudaMemcpyDeviceToHost));

    checkResult(h_from_d,h_c,SIZE);

	free(h_a);
	free(h_b);
	free(h_c);
    free(h_from_d);

	CHECK(cudaFree(d_a));
	CHECK(cudaFree(d_b));
	CHECK(cudaFree(c_dev));

    printf("vecadd[%d] Time elapsed = %g ms\n", SIZE, timer.Elapsed());
	return 0;

}