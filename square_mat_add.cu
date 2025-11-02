// nvcc square_mat_add.cu && ./a.out

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define MAX_THREADS_PER_BLOCK 1024

// Compute vector sum C = A+B
// Each thread performs one pair-wise addition
__global__
void sqmatAddKernel(float* A, float* B, float* C, int size) {
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    if(col < size) {
        for (int row = 0; row < size; row ++) {
            C[row*size + col] = A[row*size + col] + B[row*size + col];
        }
    }
}

void sqmatAdd(float* A, float* B, float* C, int n) {
    // kind of a stub function for launching a kernel
    cudaError_t err;
    int N = n*n;
    int size = N* sizeof(float);
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
    dim3 dimGrid(ceil(N/64.0), 1, 1); // host code variables
    dim3 dimBlock(32, 1, 1);
    sqmatAddKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
    // copy C from the device memory
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device vectors
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


int main() {
    float A[100][100];
    float B[100][100];
    float C[100][100];

    for (int i = 0; i < 100; i ++) {
        for (int j = 0; j < 100; j ++) {
            A[i][j] = 1.0 * i + j;
            B[i][j] = 2.0 * i + j;
        }
    }

    int n = 100;
    fprintf(stdout, "n: %d\n", n);

    sqmatAdd((float *)A, (float *)B, (float *)C, n);

    fprintf(stdout, "matrix C snapshot:\n");
    for (int i = 0; i < 10; i ++) {
        for (int j = 0; j < 10; j ++) {
            fprintf(stdout, "%lf\t", C[i][j]);
        } fprintf(stdout, "\n");
    }
}
