#include <cuda_runtime.h>
#include <stdio.h>
__global__ void checkIndex(void)
{
  printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d)\
  gridDim(%d,%d,%d)\n",threadIdx.x,threadIdx.y,threadIdx.z,
  blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,blockDim.z,
  gridDim.x,gridDim.y,gridDim.z);
  return;
}

int main(int argc,char ** argv)
{
  int nElem=1024;
  dim3 block(1024);
  dim3 grid((nElem-1)/block.x+1);
  printf("\n ================ grid.x %d block.x %d ================= \n",grid.x,block.x);
  //printf("grid.x %d grid.y %d grid.z %d\n",grid.x,grid.y,grid.z);
  //printf("block.x %d block.y %d block.z %d\n",block.x,block.y,block.z);
  checkIndex<<<grid,block>>>();

  block.x=512;
  grid.x=(nElem-1)/block.x+1;
  printf("\n ================ grid.x %d block.x %d ================= \n",grid.x,block.x);
  //printf("grid.x %d grid.y %d grid.z %d\n",grid.x,grid.y,grid.z);
  //printf("block.x %d block.y %d block.z %d\n",block.x,block.y,block.z);
  checkIndex<<<grid,block>>>();

  block.x=256;
  grid.x=(nElem-1)/block.x+1;
  printf("\n ================ grid.x %d block.x %d ================= \n",grid.x,block.x);
  //printf("grid.x %d grid.y %d grid.z %d\n",grid.x,grid.y,grid.z);
  //printf("block.x %d block.y %d block.z %d\n",block.x,block.y,block.z);
  checkIndex<<<grid,block>>>();

  block.x=128;
  grid.x=(nElem-1)/block.x+1;
  printf("\n ================ grid.x %d block.x %d ================= \n",grid.x,block.x);
  //printf("grid.x %d grid.y %d grid.z %d\n",grid.x,grid.y,grid.z);
  //printf("block.x %d block.y %d block.z %d\n",block.x,block.y,block.z);
  checkIndex<<<grid,block>>>();

  cudaDeviceReset();
  return 0;
}
