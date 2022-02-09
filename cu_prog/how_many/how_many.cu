#include <stdio.h>
// #define NUM_BLOCKS 2147483647
#define NUM_BLOCKS 1024
#define BLOCK_WIDTH 1

__global__ void hello()
{
    printf("hello world: %d\n", blockIdx.x);
}

int main(int argc, char **argv)
{
    hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();
    cudaDeviceSynchronize();
    printf("That's all!\n");
    return 0;
}
