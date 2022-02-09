#include <stdio.h>
#include "sys/time.h"

__global__
void get_hist(int *a, int *hist)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bid*blockDim.x + tid;

    // hist[(int)a[idx]] += 1;
    atomicAdd(&hist[(int)a[idx]], 1);
}

__global__ void naive_histo(int *d_bins, const int *d_in, const int BIN_COUNT)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int myItem = d_in[myId];
    int myBin = myItem % BIN_COUNT;
    d_bins[myBin]++;
}

int main()
{
    // CPU
    int size = 65536;//10240;//65536;//1024;
    int *a = new int[size];
    int length = 16;//16;//255;
    for (int i=0; i<size; ++i)
    {
        // computeBin: to which bin does this measurement belong
        // a[i] = (i*(i+1)) % length;
        a[i] = ((i/length) * ((i+1)/length))%length;
    }
    int *hist = new int[length];
    memset(hist, 0, length*sizeof(int));
    struct timeval startTime, endTime;
    gettimeofday(&startTime, NULL);
    for (int i=0; i<size; ++i)
    {
        // result[computeBin(measurements[i])]++;
        hist[(int)a[i]] += 1;
    }
    gettimeofday(&endTime, NULL);
    long int latency_cpu = (endTime.tv_sec - startTime.tv_sec)*1000000 + (endTime.tv_usec - startTime.tv_usec);    
    printf("\ncpuHistogram latency_cpu = %ld\n", latency_cpu);

    printf("\ncpu:\n");
    for (int i=0; i<length; ++i)
    {
        printf("%d ", hist[i]);
    }

    // GPU
    int *aGpu, *hGpu;
    cudaMalloc((void**)&aGpu, size*sizeof(int));
    cudaMemcpy(aGpu, a, size*sizeof(int), cudaMemcpyHostToDevice);

    int maxThreadsPerBlock = 512;
    int Dg = (size + maxThreadsPerBlock - 1)/maxThreadsPerBlock;
    cudaMalloc((void**)&hGpu, length*sizeof(int));

    gettimeofday(&startTime, NULL);
    get_hist<<<Dg, maxThreadsPerBlock>>>(aGpu, hGpu);
    gettimeofday(&endTime, NULL);
    long int latency_gpu = (endTime.tv_sec - startTime.tv_sec)*1000000 + (endTime.tv_usec - startTime.tv_usec);    
    printf("\ngpuHistogram latency_gpu = %ld\n", latency_gpu);

    cudaMemcpy(hist, hGpu, length*sizeof(int), cudaMemcpyDeviceToHost);
    printf("\ngpu:\n");
    for (int i=0; i<length; ++i)
    {
        printf("%d ", hist[i]);
    }
}