#include <stdio.h>
#include "cuda_profiler_api.h"
#include "cu_prog.h"
#include "gputimer.h"

#define SIZE 1025
#define MAX_NUM_THRE_PER_BLOCK 1024

#include <cuda_runtime.h>
//#include <iostream>
 
#include <stdio.h>
#define warpSize 32
 
__global__ void bcast(float* a, float* b) {
	int laneId = threadIdx.x & 0x1f;
	float value;

	value = a[laneId];
	// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
	// exchange a variable between threads within a warp.
	// T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
	// T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
	// T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
	// T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);
	// returns the value of var held by the thread srcLane=9 and broadcast to other 32 threads in the same warp
	value = __shfl_sync(0xffffffff, value, 9, 32);
	b[laneId] = value;
 
}
 
void printVector(char* desc, float* ptr_vec, unsigned int n){
    printf("%s =\n", desc);
 
    for(int i=0; i<n; i++){
        printf(" %5.2f ",ptr_vec[i]);
    }
 
    printf("\n");
}
 
int main() {
 
    float* a_h = NULL;
    float* a_d = NULL;
    float* b_h = NULL;
    float* b_d = NULL;
 
    a_h = (float*)malloc(warpSize*sizeof(float));
    b_h = (float*)malloc(warpSize*sizeof(float));
 
    for(int i=0; i<warpSize; i++){
        a_h[i] = i+100.0;
    }
 
    for(int i=0; i<warpSize; i++){
        b_h[i] = i+100;
    }
 
    printVector("a_h",a_h, warpSize);
    printVector("b_h",b_h, warpSize);
 
    cudaMalloc((void**)&a_d, warpSize*sizeof(float));
    cudaMalloc((void**)&b_d, warpSize*sizeof(float));
 
    cudaMemcpy(a_d, a_h, warpSize*sizeof(float), cudaMemcpyHostToDevice);  
    cudaMemcpy(b_d, b_h, warpSize*sizeof(float), cudaMemcpyHostToDevice);  
 
    bcast<<< 1, warpSize >>>(a_d, b_d);
    cudaDeviceSynchronize();
 
    cudaMemcpy(b_h, b_d, warpSize*sizeof(float), cudaMemcpyDeviceToHost);
 
    printVector("b_d", b_h, warpSize);
 
    cudaFree(a_d);
    cudaFree(b_d);
 
    return 0;

}