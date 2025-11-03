// nvcc square_mat_add.cu && ./a.out

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define MAX_THREADS_PER_BLOCK 1024
#define MAT_SIZE 2048

// Each thread performs one pair-wise addition
__global__
void sqmatAdd_1ItemPerThread_Kernel(float* A, float* B, float* C, int mat_dim) {
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    if(col < mat_dim && row < mat_dim) {
        int offset = row*mat_dim + col;
        C[offset] = A[offset] + B[offset];
    }
}

// Compute vector sum C = A+B
// Each thread performs one pair-wise addition
__global__
void sqmatAdd_1RowPerThread_Kernel(float* A, float* B, float* C, int mat_dim) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    if(row < mat_dim) {
        for (int col = 0; col < mat_dim; col ++) {
            int offset = row*mat_dim + col;
            C[offset] = A[offset] + B[offset];
        }
    }
}

// Compute vector sum C = A+B
// Each thread performs one pair-wise addition
__global__
void sqmatAdd_1ColPerThread_Kernel(float* A, float* B, float* C, int mat_dim) {
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    if(col < mat_dim) {
        for (int row = 0; row < mat_dim; row ++) {
            int offset = row*mat_dim + col;
            C[offset] = A[offset] + B[offset];
        }
    }
}

void sqmatAdd(float* A, float* B, float* C1, float* C2, float* C3, int n) {
    // kind of a stub function for launching a kernel
    cudaError_t err;
    int N = n*n;
    int size = N* sizeof(float);
    float *d_A, *d_B, *d_C;

    // Allocate device memory for A, B, and output C
    err = cudaMalloc((void **)&d_A, size);
    err = cudaMalloc((void **)&d_B, size);
    err = cudaMalloc((void **)&d_C, size);

    if (err !=cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        return;
    }

    {    
        // copy A and B to device memory
        cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
        // Kernel launch code – to have the device
        // to perform the actual vector addition
        // host code variables
        dim3 dimGrid(
            ceil((n*1.0)/32), 
            ceil((n*1.0)/32),
            1
        ); // does not work if number of blocks < 64
        dim3 dimBlock(32, 32, 1);
        fprintf(stderr, "[DEBUG] dimGrid x: %d y: %d z: %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
        fprintf(stderr, "[DEBUG] dimBlock x: %d y: %d z: %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
            
        sqmatAdd_1ItemPerThread_Kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
        // copy C from the device memory
        cudaMemcpy(C1, d_C, size, cudaMemcpyDeviceToHost);
    }

    {    
        // copy A and B to device memory
        cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
        // Kernel launch code – to have the device
        // to perform the actual vector addition
        dim3 dimGrid(1, ceil((N*1.0)/MAX_THREADS_PER_BLOCK), 1); // host code variables
        dim3 dimBlock(1, MAX_THREADS_PER_BLOCK, 1);
        fprintf(stderr, "[DEBUG] dimGrid x: %d y: %d z: %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
        fprintf(stderr, "[DEBUG] dimBlock x: %d y: %d z: %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
            
        sqmatAdd_1RowPerThread_Kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
        // copy C from the device memory
        cudaMemcpy(C2, d_C, size, cudaMemcpyDeviceToHost);
    }

    {    
        // copy A and B to device memory
        cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
        // Kernel launch code – to have the device
        // to perform the actual vector addition
        dim3 dimGrid(ceil((N*1.0)/MAX_THREADS_PER_BLOCK), 1, 1); // host code variables
        dim3 dimBlock(MAX_THREADS_PER_BLOCK, 1, 1);
        fprintf(stderr, "[DEBUG] dimGrid x: %d y: %d z: %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
        fprintf(stderr, "[DEBUG] dimBlock x: %d y: %d z: %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
            
        sqmatAdd_1ColPerThread_Kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
        // copy C from the device memory
        cudaMemcpy(C3, d_C, size, cudaMemcpyDeviceToHost);
    }

    // Free device vectors
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

float A[MAT_SIZE][MAT_SIZE];
float B[MAT_SIZE][MAT_SIZE];
float C[3][MAT_SIZE][MAT_SIZE];

int main() {

    for (int i = 0; i < MAT_SIZE; i ++) {
        for (int j = 0; j < MAT_SIZE; j ++) {
            A[i][j] = 1.0 * i + j;
            B[i][j] = 2.0 * i + j;
        }
    }

    int n = MAT_SIZE;
    fprintf(stdout, "n: %d\n", n);

    sqmatAdd((float *)A, (float *)B,
        (float *)C[0], (float *)C[1], (float *)C[2], n);

    for (int k = 0; k < 3; k ++) {
        fprintf(stdout, "matrix C%d snapshot:\n", k);
        for (int i = 0; i < 10; i ++) {
            for (int j = 0; j < 10; j ++) {
                fprintf(stdout, "%.1lf\t", C[k][i][j]);
            } fprintf(stdout, "\n");
        }
    }
}
