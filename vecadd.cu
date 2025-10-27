// nvcc vecadd.cu && ./a.out

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

// Compute vector sum C = A+B
// Each thread performs one pair-wise addition
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = (blockDim.x*blockIdx.x + threadIdx.x)*2;
    if(i<n) {
        C[i] = A[i] + B[i];
        int j = i + 1;
        if (j < n) {
            C[j] = A[j] + B[j];
        }
    }
}

void vecAdd(float* A, float* B, float* C, int n) {
    // kind of a stub function for launching a kernel
    cudaError_t err;
    int size = n* sizeof(float);
    float *d_A, *d_B, *d_C;

    // Allocate device memory for A, B, and C
    err = cudaMalloc((void **)&d_A, size);
    err = cudaMalloc((void **)&d_B, size);
    err = cudaMalloc((void **)&d_C, size);

    if (err !=cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        return;
    }
    
    // copy A and B to device memory
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    // Kernel launch code – to have the device
    // to perform the actual vector addition
    vecAddKernel<<<ceil(n/64.0), 32>>>(d_A, d_B, d_C, n);
    // copy C from the device memory
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device vectors
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


int main() {
    float A[100];
    float B[100];
    float C[100];

    for (int i = 0; i < 100; i ++) {
        A[i] = 1.0 * i;
        B[i] = 2.0 * i;
    }

    int n = sizeof(A)/sizeof(float);
    fprintf(stdout, "n: %d\n", n);

    vecAdd(A, B, C, n);

    fprintf(stdout, "all good!\n");
    for (int i = 0; i < 10; i ++) {
        fprintf(stdout, "C[%d]: %lf\n", i, C[i]);
    }
}
