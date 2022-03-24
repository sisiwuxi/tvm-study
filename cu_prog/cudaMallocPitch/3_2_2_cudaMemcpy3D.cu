#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cu_prog.h"
#include "gputimer.h"

__constant__ float constData[256];

__device__ float devData;

__device__ float* devPointer;

// Device code for cudaMemcpy3D
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr,
                         int width, int height, int depth)
{
    char* devPtr = (char*)devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0; z < depth; ++z) {
        char* slice = devPtr + z * slicePitch;
        for (int y = 0; y < height; ++y) {
            float* row = (float*)(slice + y * pitch);
            for (int x = 0; x < width; ++x) {
                float element = row[x];
            }
        }
    }
}

int main()
{
    // Host code
    int width = 64, height = 64, depth = 64;
    cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
    cudaPitchedPtr devPitchedPtr;
    cudaMalloc3D(&devPitchedPtr, extent);
    MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);

    float data[256];
    cudaMemcpyToSymbol(constData, data, sizeof(data));
    cudaMemcpyFromSymbol(data, constData, sizeof(data));
    float value = 3.14f;
    cudaMemcpyToSymbol(devData, &value, sizeof(float));
    float* ptr;
    cudaMalloc(&ptr, 256 * sizeof(float));
    cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));

    free(ptr);
	// CHECK(cudaFree(devPitchedPtr.ptr)); coredump?
}



