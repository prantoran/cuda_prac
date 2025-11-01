// nvcc query_resources.cu && ./a.out

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

int main() {
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    fprintf(stdout, "[INFO] Number of GPU devices: %d\n", dev_count);

    cudaDeviceProp dev_prop;
    for (int i = 0; i < dev_count; i++) {
        cudaGetDeviceProperties(&dev_prop, i);
        //decide if device has sufficient resources and capabilities
        fprintf(stdout, "[Info] Device: %d name: %s\n", i, dev_prop.name);
        fprintf(stdout, "\ttotalGlobalMem: %ld\n", dev_prop.totalGlobalMem);
        fprintf(stdout, "\tsharedMemPerBlock: %ld\n", dev_prop.sharedMemPerBlock);
        fprintf(stdout, "\tregsPerBlock: %d\n", dev_prop.regsPerBlock);
        fprintf(stdout, "\tmaxThreadsPerBlock: %d\n", dev_prop.maxThreadsPerBlock);
        fprintf(stdout, "\tStreaming multiprocessors count: %d\n", dev_prop.multiProcessorCount);
        fprintf(stdout, "\tClock frequency/rate: %d\n", dev_prop.clockRate);
        fprintf(stdout, "\tMax threads per block dimensions:\n");
        fprintf(stdout, "\t\tx: %d\n", dev_prop.maxThreadsDim[0]);
        fprintf(stdout, "\t\ty: %d\n", dev_prop.maxThreadsDim[1]);
        fprintf(stdout, "\t\tz: %d\n", dev_prop.maxThreadsDim[2]);
        fprintf(stdout, "\tMax blocks per grid dimensions:\n");
        fprintf(stdout, "\t\tx: %d\n", dev_prop.maxGridSize[0]);
        fprintf(stdout, "\t\ty: %d\n", dev_prop.maxGridSize[1]);
        fprintf(stdout, "\t\tz: %d\n", dev_prop.maxGridSize[2]);
    }
}
