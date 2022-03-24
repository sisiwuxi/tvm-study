#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cu_prog.h"
#include "gputimer.h"

// Device code for cudaMemcpy2D
__global__ void MyKernel(float* devPtr,
                         size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c];
        }
    }
}

int main()
{
    // Host code
    int width = 64, height = 64;
    float* devPtr;
    size_t pitch;
	CHECK(cudaMallocPitch((void**)&devPtr, &pitch, width * sizeof(int), height));
    MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

	CHECK(cudaFree(devPtr));
}

