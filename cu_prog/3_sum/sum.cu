#include <stdio.h>
#include "sys/time.h"

__global__
void sum_global(float *input, float *output, int thread)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bid*blockDim.x + tid;

    for (int i=thread/2; i>0; i>>=1)
    {
        if (idx < i)
        {
            input[idx] += input[idx + i];
        }
        __syncthreads();
    }
    if (idx == 0)
    {
        output[0] = input[0];
    }
}

__global__
void sum_shared(float *a, float *b, int thread)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bid*blockDim.x + tid;
    
    extern __shared__ float sData[];
    sData[idx] = a[idx];
    __syncthreads();
    for (int i=thread/2; i>0; i>>=1)
    {
        if (idx < i)
        {
            sData[idx] += sData[idx + i];
        }
        __syncthreads();
    }
    if (idx == 0)
    {
        b[0] = sData[0];
    }
}

void cpuSum(float *a, float *b, int thread)
{
    b[0] = 0;
    for (int i=0; i<thread; ++i)
    {
        b[0] += a[i];
    }
}

int main()
{
    int maxThreadsPerBlock = 1024;
    int thread = 8192;//102400;//8192;//8194;//4096;//1024;
    float a[thread];
    for (int i=0; i<thread; ++i)
    {
        // a[i] = i*(i+1);
        a[i] = 1.0f;
    }
    float *aGpu;
    cudaMalloc((void**)&aGpu, thread*sizeof(float));
    cudaMemcpy(aGpu, a, thread*sizeof(float), cudaMemcpyHostToDevice);

    float b[1];
    int iterations = 10000;
    struct timeval startTime, endTime;


    gettimeofday(&startTime, NULL);
    for (int i=0; i<thread; ++i)
    {
        cpuSum(a, b, thread);
    }
    gettimeofday(&endTime, NULL);
    long int latency_cpu = (endTime.tv_sec - startTime.tv_sec)*1000000 + (endTime.tv_usec - startTime.tv_usec);    
    printf("cpuSum:%f latency_cpu = %ld\n", b[0], latency_cpu);

    int Dg = (thread + maxThreadsPerBlock - 1)/maxThreadsPerBlock;

    float *bGpu_global;
    cudaMalloc((void**)&bGpu_global, 1*sizeof(float));
    gettimeofday(&startTime, NULL);
    for (int i=0; i<iterations; ++i)
    {
        cudaMemcpy(aGpu, a, thread*sizeof(float), cudaMemcpyHostToDevice);
        sum_global<<<Dg, maxThreadsPerBlock>>>(aGpu, bGpu_global, thread);
    }
    gettimeofday(&endTime, NULL);
    long int latency_global = (endTime.tv_sec - startTime.tv_sec)*1000000 + (endTime.tv_usec - startTime.tv_usec);    
    cudaMemcpy(b, bGpu_global, 1*sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum_global:%f latency_global = %ld\n", b[0], latency_global);
    

    cudaMemcpy(aGpu, a, thread*sizeof(float), cudaMemcpyHostToDevice);


    float *bGpu_shared;
    cudaMalloc((void**)&bGpu_shared, 1*sizeof(float));
    gettimeofday(&startTime, NULL);
    for (int i=0; i<iterations; ++i)
    {
        sum_shared<<<Dg, maxThreadsPerBlock, maxThreadsPerBlock*sizeof(float)>>>(aGpu, bGpu_shared, thread);
    }
    gettimeofday(&endTime, NULL);
    long int latency_shared = (endTime.tv_sec - startTime.tv_sec)*1000000 + (endTime.tv_usec - startTime.tv_usec);
    cudaMemcpy(b, bGpu_shared, 1*sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum_shared:%f latency_shared = %ld\n", b[0], latency_shared);


    return 0;
}