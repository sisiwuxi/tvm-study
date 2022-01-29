#include <cuda_runtime.h>
#include <stdio.h>
__global__ void nesthelloworld(int iSize,int iDepth)
{
    unsigned int tid=threadIdx.x;
    printf("size:%d depth:%d blockIdx:%d,threadIdx:%d\n",iSize,iDepth,blockIdx.x,threadIdx.x);
    if (iSize==1)
        return;
    int nthread=(iSize>>1);
    if (tid==0 && nthread>0)
    {
        nesthelloworld<<<1,nthread>>>(nthread,++iDepth);
        printf("-----------> nested execution size:%d depth:%d\n",iSize,iDepth);
    }

}

int main(int argc,char* argv[])
{
    //int size=64;
    int size=16;
    int block_x=2;
    dim3 block(block_x,1);
    dim3 grid((size-1)/block.x+1,1);
    nesthelloworld<<<grid,block>>>(size,0);
    cudaGetLastError();
    cudaDeviceReset();
    return 0;
}
