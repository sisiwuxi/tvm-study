#include <stdio.h>
#include "cu_prog.h"
// struct cudaTextureDesc
// {
//     enum cudaTextureAddressMode addressMode[3];
//     enum cudaTextureFilterMode  filterMode;
//     enum cudaTextureReadMode    readMode;
//     int                         sRGB;
//     int                         normalizedCoords;
//     unsigned int                maxAnisotropy;
//     enum cudaTextureFilterMode  mipmapFilterMode;
//     float                       mipmapLevelBias;
//     float                       minMipmapLevelClamp;
//     float                       maxMipmapLevelClamp;
// };
// 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// Simple transformation kernel
__global__ void transformKernel(float* output,
                                int width, int height,
                                float theta) 
{
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x / (float)width;
    float v = y / (float)height;

    // Transform coordinates
    u -= 0.5f;
    v -= 0.5f;
    float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
    float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;


    // Read from texture and write to global memory
    output[y * width + x] = tex2D(texRef, tu, tv);
}

// Host code
int main()
{
    const int height = 32;//1024;
    const int width = 32;//1024;
    float angle = 0.5;
    // Allocate and set some host data
    float *h_data = (float *)std::malloc(sizeof(float) * width * height);
    for (int i = 0; i < height * width; ++i)
        h_data[i] = i;
    printMatrix(h_data, width, height);
    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc =
               cudaCreateChannelDesc(32, 0, 0, 0,
                                     cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // Copy to device memory some data located at address h_data
    // in host memory 
    size_t size = width * height * sizeof(float);
    cudaMemcpyToArray(cuArray, 0, 0, h_data, size,
                      cudaMemcpyHostToDevice);

    // Set texture reference parameters
    texRef.addressMode[0] = cudaAddressModeWrap;
    texRef.addressMode[1] = cudaAddressModeWrap;
    texRef.filterMode     = cudaFilterModeLinear;
    texRef.normalized     = true;

    // Bind the array to the texture reference
    cudaBindTextureToArray(texRef, cuArray, channelDesc);

    // Allocate result of transformation in device memory
    float* output;
    cudaMalloc(&output, size);

    // Invoke kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);
    transformKernel<<<dimGrid, dimBlock>>>(output, width, height,
                                           angle);
    // Copy data from device back to host
    // cudaMemcpyFromArray(h_data, cuArray, 0, 0, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data, output, size, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFreeArray(cuArray);
    cudaFree(output);
    printMatrix(h_data, width, height);
    // Free host memory
    free(h_data);
    return 0;
}