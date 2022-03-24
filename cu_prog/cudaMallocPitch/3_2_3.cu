#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cu_prog.h"
#include "gputimer.h"

enum cudaLimit {
    /* other fields not shown */
    cudaLimitPersistingL2CacheSize
};

int main()
{
    // Host code
    cudaDeviceProp prop;
    int device_id = 0;
    cudaGetDeviceProperties(&prop, device_id);
    // size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
    // /* set-aside 3/4 of L2 cache for persisting accesses or the max allowed*/ 
    // cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
    size_t size = int(prop.l2CacheSize * 0.75);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);


    // Stream level attributes data structure
    cudaStreamAttrValue stream_attribute;                                         
    // Global Memory data pointer
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); 
    // Number of bytes for persistence access.
    // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
    stream_attribute.accessPolicyWindow.num_bytes = num_bytes;
    // Hint for cache hit ratio
    stream_attribute.accessPolicyWindow.hitRatio  = 0.6;
    // Type of access property on cache hit
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    // Type of access property on cache miss.
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  
    //Set the attributes to a CUDA stream of type cudaStream_t
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);  

    // Kernel level attributes data structure
    cudaKernelNodeAttrValue node_attribute;
    // Global Memory data pointer
    node_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr);
    // Number of bytes for persistence access.
    // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
    node_attribute.accessPolicyWindow.num_bytes = num_bytes;
    // Hint for cache hit ratio
    node_attribute.accessPolicyWindow.hitRatio  = 0.6;
    // Type of access property on cache hit
    node_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    // Type of access property on cache miss.
    node_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  
    //Set the attributes to a CUDA Graph Kernel node of type cudaGraphNode_t
    cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute);
    
    // This data1 is used by a kernel multiple times
    // [data1 + num_bytes) benefits from L2 persistence
    for(int i = 0; i < 10; i++) {
        cuda_kernelA<<<grid_size,block_size,0,stream>>>(data1);
    }
    // A different kernel in the same stream can also benefit
    // from the persistence of data1
    cuda_kernelB<<<grid_size,block_size,0,stream>>>(data1);
    // Setting the window size to 0 disable it
    stream_attribute.accessPolicyWindow.num_bytes = 0;
    // Overwrite the access policy attribute to a CUDA Stream
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
    // Remove any persistent lines in L2 
    cudaCtxResetPersistingL2Cache();
    // data2 can now benefit from full L2 in normal mode 
    cuda_kernelC<<<grid_size,block_size,0,stream>>>(data2);                                     
}

