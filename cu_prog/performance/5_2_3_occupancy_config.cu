#include<stdio.h>
#include<iostream>
#include "cu_prog.h"

#define SIZE 32
// include/cuda_occupancy.h
// Device code
__global__ void MyKernel(int *array, int arrayCount)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < arrayCount) {
        array[idx] *= array[idx];
    }
}

// Host code
int launchMyKernel(int *array, int arrayCount)
{
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    int blockSize; // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize; // The actual grid size needed, based on input size
    cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,(void*)MyKernel,0,arrayCount);
    // Round up according to array size
    gridSize = (arrayCount + blockSize - 1) / blockSize;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Occupancy calculator elapsed time:  %3.3f ms \n", time);

    MyKernel<<<gridSize, blockSize>>>(array, arrayCount);


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Kernel elapsed time:  %3.3f ms \n", time);

    printf("\nblockSize, minGridSize, gridSize=%d,%d,%d\n",blockSize, minGridSize, gridSize);
    cudaDeviceSynchronize();
    // If interested, the occupancy can be calculated with
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    return 0;
}

// Host code
int main()
{
    
    int *h_a; //host array 
	h_a = (int*)malloc(SIZE*sizeof(int));
    int *d_a; //device array 
    CHECK(cudaMalloc((void**)&d_a,SIZE*sizeof(int)));
    CHECK(cudaMemcpy(d_a,h_a,SIZE*sizeof(int),cudaMemcpyHostToDevice));
    launchMyKernel(d_a, SIZE);
    CHECK(cudaMemcpy(h_a,d_a,SIZE*sizeof(float),cudaMemcpyDeviceToHost));

    free(h_a);
	CHECK(cudaFree(d_a));    
    return 0;
}