#include <stdio.h>

__global__ void add(int *a, int *b, int *c, int num)
{
    int i = threadIdx.x;
    if (i < num)
    {
        c[i] = a[i] + b[i];
    }
}

int main(void)
{
    // init data on host
    int num = 10;
    int a[num], b[num], c[num];
    for (int i=0; i<num; i++)
    {
        a[i] = i;
        b[i] = i*2;
    }
    // malloc data on device
    int *ag, *bg, *cg;
    cudaMalloc((void **)&ag, num*sizeof(int));
    cudaMalloc((void **)&bg, num*sizeof(int));
    cudaMalloc((void **)&cg, num*sizeof(int));

    // copy data
    cudaMemcpy(ag, a, num*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(bg, b, num*sizeof(int), cudaMemcpyHostToDevice);

    // do
    add<<<1,num>>>(ag, bg, cg, num);

    // get data
    cudaMemcpy(c, cg, num*sizeof(int), cudaMemcpyDeviceToHost);

    // visualization
    for (int i=0; i<num; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    return 0;
}