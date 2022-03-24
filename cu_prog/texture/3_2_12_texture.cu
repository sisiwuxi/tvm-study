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


void low_level_api() {
    const texture<float, cudaTextureType2D,cudaReadModeElementType> texRef;
    const textureReference* texRefPtr;
    cudaGetTextureReference(&texRefPtr, &texRef);
    cudaChannelFormatDesc channelDesc =cudaCreateChannelDesc<float>();
    size_t offset;
    float* devPtr;
    const int height = 32;//1024;
    const int width = 32;//1024;
    const size_t pitch = width * sizeof(float);
    cudaBindTexture2D(&offset, texRefPtr, devPtr, &channelDesc,width, height, pitch);


    const texture<float, cudaTextureType2D,cudaReadModeElementType> texRef_low;
    const textureReference* texRefPtr_low;
    cudaGetTextureReference(&texRefPtr_low, &texRef_low);
    cudaChannelFormatDesc channelDesc_low;
    cudaArray_t cuArray;
    cudaGetChannelDesc(&channelDesc_low, cuArray);
    // cudaBindTextureToArray(texRef_low, cuArray, &channelDesc_low);
    cudaBindTextureToArray(texRef_low, cuArray);
}

void high_level_api()
{
    texture<float, cudaTextureType2D,cudaReadModeElementType> texRef;
    cudaChannelFormatDesc channelDesc =cudaCreateChannelDesc<float>();
    size_t offset;
    float* devPtr;
    const int height = 32;//1024;
    const int width = 32;//1024;
    const size_t pitch = width * sizeof(float);
    cudaBindTexture2D(&offset, texRef, devPtr, channelDesc,width, height, pitch);

    // texture<float, cudaTextureType2D,cudaReadModeElementType> texRef;
    cudaArray_t cuArray;
    cudaBindTextureToArray(texRef, cuArray);
}

// Simple transformation kernel
__global__ void transformKernel(float* output,
                                cudaTextureObject_t texObj,
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
    // cos(0.5) = 0.8775
    // sin(0.5) = 0.4794
    // tu = -0.5*0.8775 - (-0.5)*0.4794 + 0.5 = 0.30095000000000005
    float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
    // tv = -0.5*0.8775 + (-0.5)*0.4794 + 0.5 = -0.17845
    float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

    // Read from texture and write to global memory
    output[y * width + x] = tex2D<float>(texObj, tu, tv);
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
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // Set pitch of the source (the width in memory in bytes of the 2D array pointed
    // to by src, including padding), we dont have any padding
    const size_t spitch = width * sizeof(float);
    // Copy data located at address h_data in host memory to device memory
    cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width * sizeof(float),
                        height, cudaMemcpyHostToDevice);

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    // Allocate result of transformation in device memory
    float *output;
    cudaMalloc(&output, width * height * sizeof(float));

    // Invoke kernel
    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
                    (height + threadsperBlock.y - 1) / threadsperBlock.y);
    transformKernel<<<numBlocks, threadsperBlock>>>(output, texObj, width, height,angle);
    // Copy data from device back to host
    cudaMemcpy(h_data, output, width * height * sizeof(float),
                cudaMemcpyDeviceToHost);

    // Destroy texture object
    cudaDestroyTextureObject(texObj);

    // Free device memory
    cudaFreeArray(cuArray);
    cudaFree(output);

    printMatrix(h_data, width, height);
    // Free host memory
    free(h_data);

    return 0;
}