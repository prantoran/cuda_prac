// nvcc mat_vec_multiply.cu && ./a.out

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define MAX_THREADS_PER_BLOCK 1024
#define MAT_SIZE 2048

// Each thread performs one pair-wise addition
__global__
void sqmatvecMul_Kernel(float* A, float* B, float* C, int mat_dim) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    if(row < mat_dim) {
        float cur = 0;
        for (int col = 0; col < mat_dim; col ++) {
            cur += B[row*mat_dim + col] + C[col];
        }
        A[row] = cur;
    }
}

void sqmatvecMul(float* A, float* B, float* C, int n) {
    // kind of a stub function for launching a kernel
    cudaError_t err;
    int N = n*n;
    int size = N* sizeof(float);
    int vecsize = n*sizeof(float);
    float *d_A, *d_B, *d_C;

    // Allocate device memory for A, B, and output C
    err = cudaMalloc((void **)&d_A, vecsize);
    err = cudaMalloc((void **)&d_B, size);
    err = cudaMalloc((void **)&d_C, vecsize);

    if (err !=cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        return;
    }

    {    
        // copy A and B to device memory
        cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, C, vecsize, cudaMemcpyHostToDevice);
        // Kernel launch code – to have the device
        // to perform the actual vector addition
        // host code variables
        dim3 dimGrid(1, ceil((N*1.0)/MAX_THREADS_PER_BLOCK), 1); // host code variables
        dim3 dimBlock(1, MAX_THREADS_PER_BLOCK, 1);
        fprintf(stderr, "[DEBUG] dimGrid x: %d y: %d z: %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
        fprintf(stderr, "[DEBUG] dimBlock x: %d y: %d z: %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
            
        sqmatvecMul_Kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
        // copy C from the device memory
        cudaMemcpy(A, d_A, vecsize, cudaMemcpyDeviceToHost);
    }

    // Free device vectors
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

float A[MAT_SIZE];
float B[MAT_SIZE][MAT_SIZE];
float C[MAT_SIZE];

int main() {

    for (int i = 0; i < MAT_SIZE; i ++) {
        for (int j = 0; j < MAT_SIZE; j ++) {
            B[i][j] = 2.0 * i + j;
        }
        C[i] = i;
    }

    fprintf(stdout, "[INFO] MAT_SIZE: %d\n", MAT_SIZE);

    sqmatvecMul((float *)A, (float *)B, (float *)C, MAT_SIZE);

    fprintf(stdout, "[DEBUG] matrix A snapshot:\n");
    for (int i = 0; i < 10; i ++) {
        fprintf(stdout, "%.1lf\t", A[i]);
    } fprintf(stdout, "\n");
}
