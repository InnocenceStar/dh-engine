#ifndef CUDA_ENGINE_INTERFACE_CUH
#define CUDA_ENGINE_INTERFACE_CUH
#include <cuda_runtime.h>
namespace cuda{

enum GPU_STATUS{
	GPU_EXCUTE_GATHER_READY = 0,
	GPU_EXCUTE_GATHER_WORKING,
	GPU_EXCUTE_GATHER_DONE,
	GPU_EXCUTE_APPLY_READY,
	GPU_EXCUTE_APPLY_WORKING,
	GPU_EXCUTE_APPLY_DONE,
	GPU_EXCUTE_SCATTER_READY,
	GPU_EXCUTE_SCATTER_WORKING,
	GPU_EXCUTE_SCATTER_DONE
};

int env_init(){
	// get the nums of gpu
	int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if(error_id!=cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n ->%s\n",
              (int)error_id,cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }
    if(deviceCount==0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n",deviceCount);
    }
	int dev=0,driverVersion=0,runtimeVersion=0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Device %d:\"%s\"\n",dev,deviceProp.name);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA Driver Version / Runtime Version\t %d.%d  /  %d.%d\n",
        driverVersion/1000,(driverVersion%100)/10,
        runtimeVersion/1000,(runtimeVersion%100)/10);
    printf("CUDA Capability Major/Minor version number:\t %d.%d\n",
        deviceProp.major,deviceProp.minor);
    printf("Total amount of global memory:\t %.2f MBytes (%llu bytes)\n",
            (float)deviceProp.totalGlobalMem/pow(1024.0,3));
    printf("GPU Clock rate:\t %.0f MHz (%0.2f GHz)\n",
            deviceProp.clockRate*1e-3f,deviceProp.clockRate*1e-6f);
    printf("Memory Bus width:\t %d-bits\n", deviceProp.memoryBusWidth);
    if (deviceProp.l2CacheSize)
    {
        printf("L2 Cache Size:\t %d bytes\n",
                deviceProp.l2CacheSize);
    }
    // printf("Max Texture Dimension Size (x,y,z)\t 1D=(%d),2D=(%d,%d),3D=(%d,%d,%d)\n",
    //         deviceProp.maxTexture1D,deviceProp.maxTexture2D[0],deviceProp.maxTexture2D[1]
    //         ,deviceProp.maxTexture3D[0],deviceProp.maxTexture3D[1],deviceProp.maxTexture3D[2]);
    // printf("Max Layered Texture Size (dim) x layers\t 1D=(%d) x %d,2D=(%d,%d) x %d\n",
    //         deviceProp.maxTexture1DLayered[0],deviceProp.maxTexture1DLayered[1],
    //         deviceProp.maxTexture2DLayered[0],deviceProp.maxTexture2DLayered[1],
    //         deviceProp.maxTexture2DLayered[2]);
    // printf("Total amount of constant memory\t %lu bytes\n",
    //         deviceProp.totalConstMem);
    // printf("Total amount of shared memory per block:\t %lu bytes\n",
    //         deviceProp.sharedMemPerBlock);
    // printf("Total number of registers available per block:\t %d\n",
    //         deviceProp.regsPerBlock);
    // printf("Wrap size:\t %d\n",deviceProp.warpSize);
    // printf("Maximun number of thread per multiprocesser:\t %d\n",
    //         deviceProp.maxThreadsPerMultiProcessor);
    // printf("Maximun number of thread per block:\t %d\n",
    //         deviceProp.maxThreadsPerBlock);
    // printf("Maximun size of each dimension of a block:\t %d x %d x %d\n",
    //         deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    // printf("Maximun size of each dimension of a grid:\t %d x %d x %d\n",
    //         deviceProp.maxGridSize[0],
	//     deviceProp.maxGridSize[1],
	//     deviceProp.maxGridSize[2]);
    // printf("Maximu memory pitch\t%lu bytes\n",deviceProp.memPitch);
	return deviceCount;
}

}
#endif