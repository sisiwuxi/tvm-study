#include <stdio.h>
#include "cu_prog.h"

__global__ void conv(float *img, float *weight, float *output, int W, int H, int R, int S)
{
    int ti = threadIdx.x;
    int bi = blockIdx.x;
    int id = bi*blockDim.x + ti;
    if (id > W*H)
    {
        return;
    }
    int row = id / W;
    int col = id % W;
    for (int r=0; r<R; ++r)
    {
        for (int s=0; s<S; ++s)
        {
            float imgVal = 0.0f;
            int curRow = row - R/2 + r;
            int curCol = col - S/2 + s;
            // out of boundary
            if (curRow<0 || curCol<0 || curRow>H || curCol>W)
            {

            }
            else
            {
                // imgVal = img[id];
                imgVal = img[curRow*W + curCol];
            }
            float wVal = weight[r*S + s];
            output[id] += imgVal * wVal;
        }
    }
}

int getThreadNum()
{
    cudaDeviceProp prop;
    int count = 0;
    cudaGetDeviceCount(&count);
    printf("gpu num=%d\n", count);
    cudaGetDeviceProperties(&prop, 0);
    printf("max thread num=%d\n", prop.maxThreadsPerBlock);
    printf("max grid[%d,%d,%d]\n", prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
    return prop.maxThreadsPerBlock;
}

int main(void)
{
    // feature input
    int W = 1920;
    int H = 1080;
    float *img = new float[W*H];
    for (int row=0; row<H; ++row)
    {
        for (int col=0; col<W; ++col)
        {
            img[col + row*W] = (col+row) % 256;
        }
    }
    // weight kernel
    int R = 3;
    int S = 3;
    float *kernel = new float[R*S];
    for (int i=0; i<R*S; ++i)
    {
        kernel[i] = i%R - 1;
    }
  
    // malloc data on device
    float *imgGpu, *kernelGpu, *outputGpu;
    HANDLE_ERROR(cudaMalloc((void **)&imgGpu, W*H*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&kernelGpu, R*S*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&outputGpu, W*H*sizeof(float)));

    // copy data
    HANDLE_ERROR(cudaMemcpy(imgGpu, img, W*H*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(kernelGpu, kernel, R*S*sizeof(float), cudaMemcpyHostToDevice));

    // do
    // full threads in the device
    int threadNum = getThreadNum();
    // W*H = #output_pixel = #calculate
    int blockNum = (W*H + threadNum - 1)/threadNum;
    conv<<<blockNum,threadNum>>>
        (imgGpu, kernelGpu, outputGpu, W, H, R, S);

    // get data
    // output
    float *output = new float[W*H];    
    HANDLE_ERROR(cudaMemcpy(output, outputGpu, W*H*sizeof(float), cudaMemcpyDeviceToHost));

    // visualization
    printf("image \n");
    for (int row=0; row<10; ++row)
    {
        for (int col=0; col<10; ++col)
        {
            printf("%2.f ", img[col + row*W]);
        }
        printf("\n");
    }
    printf("kernel \n");
    for (int row=0; row<R; ++row)
    {
        for (int col=0; col<S; ++col)
        {
            printf("%2.f ", kernel[row*S+col]);
        }
        printf("\n");
    }  
    for (int row=0; row<10; ++row)
    {
        for (int col=0; col<10; ++col)
        {
            printf("%2.f ", output[col + row*W]);
        }
        printf("\n");
    }
    return 0;
}