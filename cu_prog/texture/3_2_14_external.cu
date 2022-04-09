int getCudaDeviceForVulkanPhysicalDevice(VkPhysicalDevice vkPhysicalDevice) {
    VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties = {};
    vkPhysicalDeviceIDProperties.sType =
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    vkPhysicalDeviceIDProperties.pNext = NULL;
    VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2 = {};
    vkPhysicalDeviceProperties2.sType =
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;
    vkGetPhysicalDeviceProperties2(vkPhysicalDevice, &vkPhysicalDeviceProperties2);
    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);
    for (int cudaDevice = 0; cudaDevice < cudaDeviceCount; cudaDevice++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, cudaDevice);
        if (!memcmp(&deviceProp.uuid, vkPhysicalDeviceIDProperties.deviceUUID, VK_UUID_SIZE)) {
            return cudaDevice;
        }
    }
    return cudaInvalidDeviceId;
}

cudaExternalMemory_t importVulkanMemoryObjectFromFileDescriptor(int fd, unsigned long long size, bool isDedicated) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = fd;
    desc.size = size;
    if (isDedicated) {
        desc.flags |= cudaExternalMemoryDedicated;
    }
    cudaImportExternalMemory(&extMem, &desc);
    // Input parameter 'fd' should not be used beyond this point as CUDA has assumed ownership of it
    return extMem;
}

cudaExternalMemory_t importVulkanMemoryObjectFromNTHandle(HANDLE handle, unsigned long long size, bool isDedicated) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    desc.handle.win32.handle = handle;
    desc.size = size;
    if (isDedicated) {
        desc.flags |= cudaExternalMemoryDedicated;
    }
    cudaImportExternalMemory(&extMem, &desc);
    // Input parameter 'handle' should be closed if it's not needed anymore
    CloseHandle(handle);
    return extMem;
}

cudaExternalMemory_t importVulkanMemoryObjectFromNamedNTHandle(LPCWSTR
 name, unsigned long long size, bool isDedicated) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    desc.handle.win32.name = (void *)name;
    desc.size = size;
    if (isDedicated) {
        desc.flags |= cudaExternalMemoryDedicated;
    }
    cudaImportExternalMemory(&extMem, &desc);
    return extMem;
}


cudaExternalMemory_t importVulkanMemoryObjectFromKMTHandle(HANDLE handle, unsigned long long size, bool isDedicated) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
    desc.handle.win32.handle = (void *)handle;
    desc.size = size;
    if (isDedicated) {
    desc.flags |= cudaExternalMemoryDedicated;
    }
    cudaImportExternalMemory(&extMem, &desc);
    return extMem;
}

void * mapBufferOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, unsigned long long size) {
    void *ptr = NULL;
    cudaExternalMemoryBufferDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.offset = offset;
    desc.size = size;
    cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);
    // Note: ‘ptr’ must eventually be freed using cudaFree()

    return ptr;
}

cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, cudaChannelFormatDesc *formatDesc, cudaExtent *extent, unsigned int flags, unsigned int numLevels) {
    cudaMipmappedArray_t mipmap = NULL;
    cudaExternalMemoryMipmappedArrayDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.offset = offset;
    desc.formatDesc = *formatDesc;
    desc.extent = *extent;
    desc.flags = flags;
    desc.numLevels = numLevels;
    // Note: ‘mipmap’ must eventually be freed using cudaFreeMipmappedArray()
    cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);
    return mipmap;
}
cudaChannelFormatDesc getCudaChannelFormatDescForVulkanFormat(VkFormat format)
{
    cudaChannelFormatDesc d;
    memset(&d, 0, sizeof(d));
    switch (format) {
        case VK_FORMAT_R8_UINT: d.x = 8; d.y = 0; d.z = 0; d.w = 0; d.f= cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R8_SINT: d.x = 8; d.y = 0; d.z = 0; d.w = 0; d.f= cudaChannelFormatKindSigned; break;
        case VK_FORMAT_R8G8_UINT: d.x = 8; d.y = 8; d.z = 0; d.w = 0; d.f= cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R8G8_SINT: d.x = 8; d.y = 8; d.z = 0; d.w = 0; d.f= cudaChannelFormatKindSigned; break;
        case VK_FORMAT_R8G8B8A8_UINT: d.x = 8; d.y = 8; d.z = 8; d.w = 8; d.f= cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R8G8B8A8_SINT: d.x = 8; d.y = 8; d.z = 8; d.w = 8; d.f= cudaChannelFormatKindSigned; break;
        case VK_FORMAT_R16_UINT: d.x = 16; d.y = 0; d.z = 0; d.w = 0; d.f= cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R16_SINT: d.x = 16; d.y = 0; d.z = 0; d.w = 0; d.f= cudaChannelFormatKindSigned; break;
        case VK_FORMAT_R16G16_UINT: d.x = 16; d.y = 16; d.z = 0; d.w = 0; d.f= cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R16G16_SINT: d.x = 16; d.y = 16; d.z = 0; d.w = 0; d.f= cudaChannelFormatKindSigned; break;
        case VK_FORMAT_R16G16B16A16_UINT: d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f= cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R16G16B16A16_SINT: d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f= cudaChannelFormatKindSigned; break;
        case VK_FORMAT_R32_UINT: d.x = 32; d.y = 0; d.z = 0; d.w = 0; d.f= cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R32_SINT: d.x = 32; d.y = 0; d.z = 0; d.w = 0; d.f= cudaChannelFormatKindSigned; break;
        case VK_FORMAT_R32_SFLOAT: d.x = 32; d.y = 0; d.z = 0; d.w = 0; d.f= cudaChannelFormatKindFloat; break;
        case VK_FORMAT_R32G32_UINT: d.x = 32; d.y = 32; d.z = 0; d.w = 0; d.f= cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R32G32_SINT: d.x = 32; d.y = 32; d.z = 0; d.w = 0; d.f= cudaChannelFormatKindSigned; break;
        case VK_FORMAT_R32G32_SFLOAT: d.x = 32; d.y = 32; d.z = 0; d.w = 0; d.f= cudaChannelFormatKindFloat; break;
        case VK_FORMAT_R32G32B32A32_UINT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f= cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R32G32B32A32_SINT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f= cudaChannelFormatKindSigned; break;
        case VK_FORMAT_R32G32B32A32_SFLOAT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f= cudaChannelFormatKindFloat; break;
        default: assert(0);
    }

 return d;
}

cudaExtent getCudaExtentForVulkanExtent(VkExtent3D vkExt, uint32_t arrayLayers,
 VkImageViewType vkImageViewType) {
 cudaExtent e = { 0, 0, 0 };
 switch (vkImageViewType) {
 case VK_IMAGE_VIEW_TYPE_1D: e.width = vkExt.width; e.height = 0;
 e.depth = 0; break;
 case VK_IMAGE_VIEW_TYPE_2D: e.width = vkExt.width; e.height =
 vkExt.height; e.depth = 0; break;
 case VK_IMAGE_VIEW_TYPE_3D: e.width = vkExt.width; e.height =
 vkExt.height; e.depth = vkExt.depth; break;
 case VK_IMAGE_VIEW_TYPE_CUBE: e.width = vkExt.width; e.height =
 vkExt.height; e.depth = arrayLayers; break;
 case VK_IMAGE_VIEW_TYPE_1D_ARRAY: e.width = vkExt.width; e.height = 0;
 e.depth = arrayLayers; break;
 case VK_IMAGE_VIEW_TYPE_2D_ARRAY: e.width = vkExt.width; e.height =
 vkExt.height; e.depth = arrayLayers; break;
 case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: e.width = vkExt.width; e.height =
 vkExt.height; e.depth = arrayLayers; break;
 default: assert(0);
 }
 return e;
}
unsigned int getCudaMipmappedArrayFlagsForVulkanImage(VkImageViewType
 vkImageViewType, VkImageUsageFlags vkImageUsageFlags, bool allowSurfaceLoadStore) {
    unsigned int flags = 0;
    switch (vkImageViewType) {
        case VK_IMAGE_VIEW_TYPE_CUBE: flags |= cudaArrayCubemap;
            break;
        case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: flags |= cudaArrayCubemap |cudaArrayLayered; break;
        case VK_IMAGE_VIEW_TYPE_1D_ARRAY: flags |= cudaArrayLayered;
            break;
        case VK_IMAGE_VIEW_TYPE_2D_ARRAY: flags |= cudaArrayLayered;
            break;
        default: break;
    }
    if (vkImageUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
        flags |= cudaArrayColorAttachment;
    }
    if (allowSurfaceLoadStore) {
        flags |= cudaArraySurfaceLoadStore;
    }
    return flags;
}


//  synchronization objects


cudaExternalSemaphore_t importVulkanSemaphoreObjectFromFileDescriptor(int fd) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    desc.handle.fd = fd;
    cudaImportExternalSemaphore(&extSem, &desc);
    // Input parameter 'fd' should not be used beyond this point as CUDA has assumed ownership of it
    return extSem;
}



cudaExternalSemaphore_t importVulkanSemaphoreObjectFromNTHandle(HANDLE handle) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
    desc.handle.win32.handle = handle;
    cudaImportExternalSemaphore(&extSem, &desc);
    // Input parameter 'handle' should be closed if it's not needed anymore
    CloseHandle(handle);
    return extSem;
}

cudaExternalSemaphore_t importVulkanSemaphoreObjectFromNamedNTHandle(LPCWSTR name) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
    desc.handle.win32.name = (void *)name;
    cudaImportExternalSemaphore(&extSem, &desc);
    return extSem;
}


cudaExternalSemaphore_t importVulkanSemaphoreObjectFromKMTHandle(HANDLE handle)
{
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
    desc.handle.win32.handle = (void *)handle;
    cudaImportExternalSemaphore(&extSem, &desc);
    return extSem;
}

void signalExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream) {
    cudaExternalSemaphoreSignalParams params = {};
    memset(&params, 0, sizeof(params));
    cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

void waitExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream)
{
    cudaExternalSemaphoreWaitParams params = {};
    memset(&params, 0, sizeof(params));
    cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream);
}
