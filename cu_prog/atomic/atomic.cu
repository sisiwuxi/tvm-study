#include <stdio.h>
#include "gputimer.h"
 
#define NUM_THREADS 1000000
#define ARRAY_SIZE  100
 
#define BLOCK_WIDTH 1000
 
void print_array(int *array, int size)
{
    printf("{ ");
    for (int idx = 0; idx < size; idx++)  { printf("%d ", array[idx]); }
    printf("}\n");
}
 
__global__ void increment_naive(int *g)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
	int idx = bid * blockDim.x + tid; 
 
	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	idx = idx % ARRAY_SIZE;  
	g[idx] = g[idx] + 1;
}
 
__global__ void increment_atomic(int *g)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
	int idx = bid * blockDim.x + tid; 
 
	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	idx = idx % ARRAY_SIZE;  
	atomicAdd(&g[idx], 1);
}
 
int main(int argc,char **argv)
{   
    GpuTimer timer;
    printf("%d total threads in %d blocks writing into %d array elements\n",
           NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, ARRAY_SIZE);
 
    // declare and allocate host memory
    int h_array[ARRAY_SIZE];
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
 
    // declare, allocate, and zero out GPU memory
    int * d_array;
    cudaMalloc((void **) &d_array, ARRAY_BYTES);
    cudaMemset((void *) d_array, 0, ARRAY_BYTES);
 
    // launch the kernel - comment out one of these
    timer.Start();
    // increment_naive<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
    increment_atomic<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
    timer.Stop();
    
    // copy back the array of sums from GPU and print
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    print_array(h_array, ARRAY_SIZE);
    printf("Time elapsed = %g ms\n", timer.Elapsed());
 
    // free GPU memory allocation and exit
    cudaFree(d_array);
    return 0;
}