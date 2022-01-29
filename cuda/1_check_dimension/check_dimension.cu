#include <cuda_runtime.h>
#include <stdio.h>
__global__ void checkIndex(void)
{
  printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d)\
  gridDim(%d,%d,%d)\n",threadIdx.x,threadIdx.y,threadIdx.z,
  blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,blockDim.z,
  gridDim.x,gridDim.y,gridDim.z);
}
int main(int argc,char **argv)
{
  int nElem=6;
  dim3 block(3);
  dim3 grid((nElem+block.x-1)/block.x);
  printf("grid.x %d grid.y %d grid.z %d\n",grid.x,grid.y,grid.z);
  printf("block.x %d block.y %d block.z %d\n",block.x,block.y,block.z);
  checkIndex<<<grid,block>>>();
  cudaDeviceSynchronize();
  dim3 block2(2);
  dim3 grid2((nElem+block.x-1)/block.x);
  printf("grid2.x %d grid2.y %d grid2.z %d\n",grid2.x,grid2.y,grid2.z);
  printf("block2.x %d block2.y %d block2.z %d\n",block2.x,block2.y,block2.z);
  checkIndex<<<grid2,block2>>>();
  cudaDeviceReset();
  return 0;
}
