// 1. set device
// 2. malloc
// 3. memcpy from host to device
// 4. kernel
// 5. memcpy from device to host
// 6. free
// 7. reset device

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <iostream>

__global__ void kernelFunc(float * a)
{
    a[threadIdx.x] = 1;
}

int main_v0(int argc, char **argv)
{
    // 1. set device
    cudaSetDevice(0);
    // 2. malloc
    float *aGPU;
    cudaMalloc((void**)&aGPU, 16*sizeof(float));
    float aCPU[16] = {0};
    // 3. memcpy from host to device
    cudaMemcpy(aGPU, aCPU, 16*sizeof(float), cudaMemcpyHostToDevice);
    // 4. kernel
    kernelFunc<<<1,16>>>(aGPU);
    // 5. memcpy from device to host
    cudaMemcpy(aCPU, aGPU, 16*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0; i<16; i++)
    {
        printf("%f ",aCPU[i]);
    }
    // 6. free
    cudaFree(aGPU);
    // 7. reset device
    cudaDeviceReset();
    return 0;
}

static void HandleError(cudaError_t err,
                        const char *file,
                        int line) 
                        {
                            if (err != cudaSuccess)
                            {
                                printf("%s in %s at line %d\n",
                                cudaGetErrorString(err),
                                file, line);
                                exit(EXIT_FAILURE);
                            }
                        }
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

int main(int argc, char **argv)
{
    // 1. set device
    // get info
    int gpuCount = -1;
    cudaGetDeviceCount(&gpuCount);
    printf("gpuCount=%d\n", gpuCount);
    if (gpuCount < 0) 
    {
        printf("no device\n");
        return -1;
    }
    cudaSetDevice(gpuCount - 1);
    int deviceId;
    cudaGetDevice(&deviceId);
    printf("deviceId=%d\n", deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    printf("maxThreadsPerBlock:%d\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim:%d\n", prop.maxThreadsDim[0]);
    printf("maxGridSize:%d\n", prop.maxGridSize[0]);
    printf("totalConstMem:%d\n", prop.totalConstMem);
    printf("clockRate:%d\n", prop.clockRate);
    printf("integrated:%d\n", prop.integrated);
    // recommend device
    cudaChooseDevice(&deviceId, &prop);
    printf("recommend deviceId=%d\n", deviceId);
    // multi device
    int deviceList[2] = {0,1};
    HANDLE_ERROR(cudaSetValidDevices(deviceList, 1));
    // HANDLE_ERROR(cudaSetValidDevices(deviceList, 2));
    // int deviceList[2] = {1,0};
    // HANDLE_ERROR(cudaSetValidDevices(deviceList, 1));
    // 2. malloc
    float *aGPU;
    cudaMalloc((void**)&aGPU, 16*sizeof(float));
    float aCPU[16] = {0};
    // 3. memcpy from host to device
    cudaMemcpy(aGPU, aCPU, 16*sizeof(float), cudaMemcpyHostToDevice);
    // 4. kernel
    kernelFunc<<<1,16>>>(aGPU);
    // 5. memcpy from device to host
    cudaMemcpy(aCPU, aGPU, 16*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0; i<16; i++)
    {
        printf("%f ",aCPU[i]);
    }
    // 6. free
    cudaFree(aGPU);
    // 7. reset device
    cudaDeviceReset();
    return 0;
}