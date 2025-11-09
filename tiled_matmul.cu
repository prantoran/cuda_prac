// nvcc tiled_matmul.cu && ./a.out
// Shared memory and number of registers per SM can be a limiting factor

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define MAT_SIZE 2048
#define TILE_WIDTH 32
#define MAX_THREADS_PER_BLOCK TILE_WIDTH*TILE_WIDTH

__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int J, int K, int L) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    // Identify the row and column of the d_P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;
    // Loop over the d_M and d_N tiles required to compute d_P element
    for (int ph = 0; ph < ceil(1.0*K/TILE_WIDTH); ++ph) {
        // Collaborative loading of d_M and d_N tiles into shared memory
        if ((Row < J) && (ph*TILE_WIDTH + tx < K))
            Mds[ty][tx] = d_M[Row*K + (ph*TILE_WIDTH + tx)];
        if ((ph*TILE_WIDTH + ty < K) && Col < L)
            Nds[ty][tx] = d_N[(ph*TILE_WIDTH + ty)*L + Col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    if (Row < J && Col < L)
        d_P[Row*L + Col] = Pvalue;
}

void matmul(float* A, float* B, float* C, int J, int K, int L) {
    // kind of a stub function for launching a kernel
    cudaError_t err;
    int sizeA = J*(K*sizeof(float));
    int sizeB = K*(L*sizeof(float));
    int sizeC = J*(L*sizeof(float));
    float *d_A, *d_B, *d_C;

    // Allocate device memory for A, B, and output C
    err = cudaMalloc((void **)&d_A, sizeA);
    err = cudaMalloc((void **)&d_B, sizeB);
    err = cudaMalloc((void **)&d_C, sizeC);

    if (err !=cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        return;
    }

    {    
        // copy A and B to device memory
        cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
        // Kernel launch code – to have the device
        // to perform the actual vector addition
        dim3 dimGrid(ceil((1.0*J)/TILE_WIDTH), ceil((1.0*L)/TILE_WIDTH), 1); // host code variables
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        fprintf(stderr, "[DEBUG] dimGrid x: %d y: %d z: %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
        fprintf(stderr, "[DEBUG] dimBlock x: %d y: %d z: %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
            
        MatrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, J, K, L);
        // copy C from the device memory
        cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    }

    // Free device vectors
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

float A[2*MAT_SIZE][MAT_SIZE];
float B[MAT_SIZE][3*MAT_SIZE];
float C[2*MAT_SIZE][3*MAT_SIZE];

int main() {
    for (int i = 0; i < 2*MAT_SIZE; i ++) {
        for (int j = 0; j < MAT_SIZE; j ++) {
            A[i][j] = 0;
        }
        A[i][i] = 1;
    }
    for (int i = 0; i < MAT_SIZE; i ++) {
        for (int j = 0; j < 2*MAT_SIZE; j ++) {
            B[i][j] = 0;
        }
        B[i][i] = 2;
    }
    int n = MAT_SIZE;
    fprintf(stdout, "n: %d\n", n);

    matmul((float *)A, (float *)B, (float *)C, MAT_SIZE, MAT_SIZE, MAT_SIZE);

    fprintf(stdout, "matrix C snapshot:\n");
    for (int i = 30; i < 45; i ++) {
        for (int j = 30; j < 45; j ++) {
            fprintf(stdout, "%.1lf\t", C[i][j]);
        } fprintf(stdout, "\n");
    }
}
