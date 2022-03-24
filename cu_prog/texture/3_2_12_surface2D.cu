#include <stdio.h>
#include "cu_prog.h"
// 2D surfaces
surface<void, 2> inputSurfRef;
surface<void, 2> outputSurfRef;
            
// Simple copy kernel
__global__ void copyKernel(int width, int height) 
{
    // Calculate surface coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        uchar4 data;
        // Read from input surface
        surf2Dread(&data,  inputSurfRef, x * 4, y);
        // Write to output surface
        surf2Dwrite(data, outputSurfRef, x * 4, y);
    }
}

// Host code
int main()
{
    const int height = 32;//1024;
    const int width = 32;//1024;

    // Allocate and set some host data
    unsigned char *h_data =
        (unsigned char *)std::malloc(sizeof(unsigned char) * width * height * 4);
    for (int i = 0; i < height * width * 4; ++i)
        h_data[i] = i;
    printSurface(h_data, width, height, 4);
    // Allocate CUDA arrays in device memory
    cudaChannelFormatDesc channelDesc =
             cudaCreateChannelDesc(8, 8, 8, 8,
                                   cudaChannelFormatKindUnsigned);
    cudaArray* cuInputArray;
    cudaMallocArray(&cuInputArray, &channelDesc, width, height,
                    cudaArraySurfaceLoadStore);
    cudaArray* cuOutputArray;
    cudaMallocArray(&cuOutputArray, &channelDesc, width, height,
                    cudaArraySurfaceLoadStore);

    // Copy to device memory some data located at address h_data
    // in host memory
    size_t size = 4 * width * sizeof(unsigned char) * height;
    cudaMemcpyToArray(cuInputArray, 0, 0, h_data, size,cudaMemcpyHostToDevice);

    // Bind the arrays to the surface references
    cudaBindSurfaceToArray(inputSurfRef, cuInputArray);
    cudaBindSurfaceToArray(outputSurfRef, cuOutputArray);

    // Invoke kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);
    copyKernel<<<dimGrid, dimBlock>>>(width, height);

    // Free device memory
    cudaFreeArray(cuInputArray);
    cudaFreeArray(cuOutputArray);
    // Copy data from device back to host
    cudaMemcpyFromArray(h_data, cuOutputArray, 0, 0, size, cudaMemcpyDeviceToHost);
    // Free device memory
    printSurface(h_data, width, height, 4);
    return 0;
}
