#include<stdio.h>

__global__ void MyKernel(float *c, float *a, const int size)
{
    return;
}

// CPU callback
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *data){
    printf("Inside callback %d\n", (size_t)data);
    return;
}

void create_graph() {
    /*
            A
          /   \
        B       C
          \   /
            D
    */

    // // Create the graph - it starts out empty
    // cudaGraphCreate(&graph, 0);

    // // For the purpose of this example, we'll create
    // // the nodes separately from the dependencies to
    // // demonstrate that it can be done in two stages.
    // // Note that dependencies can also be specified 
    // // at node creation. 
    // cudaGraphAddKernelNode(&a, graph, NULL, 0, &nodeParams);
    // cudaGraphAddKernelNode(&b, graph, NULL, 0, &nodeParams);
    // cudaGraphAddKernelNode(&c, graph, NULL, 0, &nodeParams);
    // cudaGraphAddKernelNode(&d, graph, NULL, 0, &nodeParams);

    // // Now set up dependencies on each node
    // cudaGraphAddDependencies(graph, &a, &b, 1);     // A->B
    // cudaGraphAddDependencies(graph, &a, &c, 1);     // A->C
    // cudaGraphAddDependencies(graph, &b, &d, 1);     // B->D
    // cudaGraphAddDependencies(graph, &c, &d, 1);     // C->D

    // cudaGraph_t graph;
    // cudaStreamBeginCapture(stream);
    // kernel_A<<< ..., stream >>>(...);
    // kernel_B<<< ..., stream >>>(...);
    // libraryCall(stream);
    // kernel_C<<< ..., stream >>>(...);

    // cudaStreamEndCapture(stream, &graph);
    // cudaStreamIsCapturing()

    // // stream1 is the origin stream
    // cudaStreamBeginCapture(stream1);
    // kernel_A<<< ..., stream1 >>>(...);
    // // Fork into stream2
    // cudaEventRecord(event1, stream1);
    // cudaStreamWaitEvent(stream2, event1);
    // kernel_B<<< ..., stream1 >>>(...);
    // kernel_C<<< ..., stream2 >>>(...);
    // // Join stream2 back to origin stream (stream1)
    // cudaEventRecord(event2, stream2);
    // cudaStreamWaitEvent(stream1, event2);
    // kernel_D<<< ..., stream1 >>>(...);
    // // End capture in the origin stream
    // cudaStreamEndCapture(stream1, &graph);
    // // stream1 and stream2 no longer in capture mode 
    return;
}

// void update_graph() {
//     cudaGraphExec_t graphExec = NULL;

//     for (int i = 0; i < 10; i++) {
//         cudaGraph_t graph;
//         cudaGraphExecUpdateResult updateResult;
//         cudaGraphNode_t errorNode;

//         // In this example we use stream capture to create the graph.
//         // You can also use the Graph API to produce a graph.
//         cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

//         // Call a user-defined, stream based workload, for example
//         do_cuda_work(stream);

//         cudaStreamEndCapture(stream, &graph);

//         // If we've already instantiated the graph, try to update it directly
//         // and avoid the instantiation overhead
//         if (graphExec != NULL) {
//             // If the graph fails to update, errorNode will be set to the
//             // node causing the failure and updateResult will be set to a
//             // reason code.
//             cudaGraphExecUpdate(graphExec, graph, &errorNode, &updateResult);
//         }

//         // Instantiate during the first iteration or whenever the update
//         // fails for any reason
//         if (graphExec == NULL || updateResult != cudaGraphExecUpdateSuccess) {

//             // If a previous update failed, destroy the cudaGraphExec_t
//             // before re-instantiating it
//             if (graphExec != NULL) {
//                 cudaGraphExecDestroy(graphExec);
//             }   
//             // Instantiate graphExec from graph. The error node and
//             // error message parameters are unused here.
//             cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
//         }   

//         cudaGraphDestroy(graph);
//         cudaGraphLaunch(graphExec, stream);
//         cudaStreamSynchronize(stream);
//     }
// }

void update_node_update() {
    // cudaGraphExecKernelNodeSetParams()
    // cudaGraphExecMemcpyNodeSetParams()
    // cudaGraphExecMemsetNodeSetParams()
    // cudaGraphExecHostNodeSetParams()
    // cudaGraphExecChildGraphNodeSetParams()
    // cudaGraphExecEventRecordNodeSetEvent()
    // cudaGraphExecEventWaitNodeSetEvent()
    // cudaGraphExecExternalSemaphoresSignalNodeSetParams()
    // cudaGraphExecExternalSemaphoresWaitNodeSetParams()

}

void event() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    float* inputDev, *outputDev;
    float* inputHost, *outputHost;
    int size = 16;
    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i) {
        cudaMemcpyAsync(inputDev + i * size, inputHost + i * size,
                        size, cudaMemcpyHostToDevice, stream[i]);
        MyKernel<<<100, 512, 0, stream[i]>>>
                (outputDev + i * size, inputDev + i * size, size);
        cudaMemcpyAsync(outputHost + i * size, outputDev + i * size,
                        size, cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void multiDevice() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n",
            device, deviceProp.major, deviceProp.minor);
    }

    size_t size = 1024 * sizeof(float);
    cudaSetDevice(0);            // Set device 0 as current
    float* p0;
    cudaMalloc(&p0, size);       // Allocate memory on device 0
    MyKernel<<<1000, 128>>>(p0,p0,size); // Launch kernel on device 0
    cudaSetDevice(1);            // Set device 1 as current
    float* p1;
    cudaMalloc(&p1, size);       // Allocate memory on device 1
    MyKernel<<<1000, 128>>>(p1,p1,size); // Launch kernel on device 1


    cudaSetDevice(0);               // Set device 0 as current
    cudaStream_t s0;
    cudaStreamCreate(&s0);          // Create stream s0 on device 0
    MyKernel<<<100, 64, 0, s0>>>(p0,p0,size); // Launch kernel on device 0 in s0
    cudaSetDevice(1);               // Set device 1 as current
    cudaStream_t s1;
    cudaStreamCreate(&s1);          // Create stream s1 on device 1
    MyKernel<<<100, 64, 0, s1>>>(p0,p0,size); // Launch kernel on device 1 in s1

    // This kernel launch will fail:
    MyKernel<<<100, 64, 0, s0>>>(p0,p0,size); // Launch kernel on device 1 in s0

    // cudaEventRecord
    // cudaEventElapsedTime
    // cudaEventSynchronize
    // cudaEventQuery
    // cudaStreamWaitEvent

    cudaSetDevice(0);                   // Set device 0 as current
    // float* p0;
    // size_t size = 1024 * sizeof(float);
    cudaMalloc(&p0, size);              // Allocate memory on device 0
    MyKernel<<<1000, 128>>>(p0,p0,size);        // Launch kernel on device 0
    cudaSetDevice(1);                   // Set device 1 as current
    cudaDeviceEnablePeerAccess(0, 0);   // Enable peer-to-peer access
                                        // with device 0

    // Launch kernel on device 1
    // This kernel launch can access memory on device 0 at address p0
    MyKernel<<<1000, 128>>>(p0,p0,size);


    cudaSetDevice(0);                   // Set device 0 as current
    // float* p0;
    // size_t size = 1024 * sizeof(float);
    cudaMalloc(&p0, size);              // Allocate memory on device 0
    cudaSetDevice(1);                   // Set device 1 as current
    // float* p1;
    cudaMalloc(&p1, size);              // Allocate memory on device 1
    cudaSetDevice(0);                   // Set device 0 as current
    MyKernel<<<1000, 128>>>(p0,p0,size);        // Launch kernel on device 0
    cudaSetDevice(1);                   // Set device 1 as current
    cudaMemcpyPeer(p1, 1, p0, 0, size); // Copy p0 to p1
    MyKernel<<<1000, 128>>>(p0,p0,size);        // Launch kernel on device 1


}







int main()
{
    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i) {
        cudaStreamCreate(&stream[i]);
    }
        
    float* hostPtr;
    int size = 16;
    cudaMallocHost(&hostPtr, 2 * size);
    float* inputDevPtr;
    float* outputDevPtr;

    for (int i = 0; i < 2; ++i) {
        cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size, size, cudaMemcpyHostToDevice, stream[i]);
        MyKernel <<<100, 512, 0, stream[i]>>>
            (outputDevPtr + i * size, inputDevPtr + i * size, size);
        cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                        size, cudaMemcpyDeviceToHost, stream[i]);
    }
    // overlap
    for (int i = 0; i < 2; ++i) {
        cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size, size, cudaMemcpyHostToDevice, stream[i]);
    }
    for (int i = 0; i < 2; ++i)
    {
        MyKernel<<<100, 512, 0, stream[i]>>>
            (outputDevPtr + i * size, inputDevPtr + i * size, size);
        for (int i = 0; i < 2; ++i)
        cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                        size, cudaMemcpyDeviceToHost, stream[i]);
    }

    float* devPtrIn[2];
    float* hostPtrIn[2];
    float* devPtrOut[2];
    for (size_t i = 0; i < 2; ++i) {
        cudaMemcpyAsync(devPtrIn[i], hostPtrIn[i], size, cudaMemcpyHostToDevice, stream[i]);
        MyKernel<<<100, 512, 0, stream[i]>>>(devPtrOut[i], devPtrIn[i], size);
        cudaMemcpyAsync(hostPtrIn[i], devPtrOut[i], size, cudaMemcpyDeviceToHost, stream[i]);
        // cudaLaunchHostFunc(stream[i], MyCallback, (void*)i);
    }

    // get the range of stream priorities for this device
    int priority_high, priority_low;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    // create streams with highest and lowest available priorities
    cudaStream_t st_high, st_low;
    cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high);
    cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, priority_low);

    for (int i = 0; i < 2; ++i)
    {
        cudaStreamDestroy(stream[i]);
    }
    cudaDeviceSynchronize();
    cudaEvent_t event = 0;
    unsigned int flag = 0;
    cudaStreamWaitEvent(st_high, event, flag);
    cudaStreamQuery(st_high);
}

