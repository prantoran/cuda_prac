// nvcc img_blur.cu -lpng && ./a.out p.png q.png

#include <stdio.h>
#include "lib/png.h"

#define CHANNELS 4 // rgb + alpha
#define BLUR_SIZE 9

// we have 3 channels corresponding to RGB
// The input image is encoded as unsigned characters [0, 255]
__global__
void blurKernel(
    unsigned char * Pout, unsigned char * Pin,
    int width, int height
) {
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if (Col < width && Row < height) {
        int pixVals[CHANNELS];
        int pixels[CHANNELS];
        for (int ch = 0; ch < CHANNELS; ch ++) {
            pixVals[ch] = 0;
            pixels[ch] = 0;
        }
        // Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box
        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; blurRow ++) {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; blurCol ++) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;
                if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                    int rgbaOffset = (curRow*width + curCol)*CHANNELS;
                    for (int ch = 0; ch < CHANNELS; ch ++) {
                        pixVals[ch] += Pin[rgbaOffset + ch];
                        pixels[ch] ++;
                    }
                }
            }
        }
        int rgbaOffset_target = (Row*width + Col)*CHANNELS;
        for (int ch = 0; ch < CHANNELS; ch ++) {
            Pout[rgbaOffset_target + ch] = (unsigned char)(pixVals[ch] / pixels[ch]);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.png> <grayscale_output.png>\n", argv[0]);
        return 1;
    }
    const char * filename = argv[1];
    const char * output_filename = argv[2];

    Image img = {0};

    if (!read_png_file(filename, &img)) {
        fprintf(stderr, "Failed to read PNG file.\n");
        return 1;
    }

    printf("Width: %u, Height: %u\n", img.width, img.height);

    // Example: Access first pixel (RGBA)
    printf("First pixel RGBA: %u %u %u %u\n",
           img.data[0], img.data[1], img.data[2], img.data[3]);

    int img_width = img.width;
    int img_height = img.height;
    int size = img_width*img_height;

    // check if all zero
    bool non_zero_found = false;
    for (int i = 0; !non_zero_found && i < size*CHANNELS; i ++) {
        if (img.data[i] > 0)
            non_zero_found = true;
    }
    if (non_zero_found) {
        fprintf(stderr, "[DEBUG] Non-zero pixels found in input image.\n");
    } else {
        fprintf(stderr, "[ERROR] No non-zero pixels found in input image\n");
    }

    unsigned char * out_png = (unsigned char *)malloc(size*CHANNELS);

    unsigned char * d_Pin, * d_Pout;
    cudaError_t err;

    // Allocate device memory for Pin and Pout
    err = cudaMalloc((void **)&d_Pin,  size*CHANNELS);
    err = cudaMalloc((void **)&d_Pout, size*CHANNELS);
    if (err !=cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        return;
    }

     // copy Pin to device memory
    cudaMemcpy(d_Pin, img.data, size*CHANNELS, cudaMemcpyHostToDevice);

    // Kernel launch code – to have the device
    // to perform the actual vector addition
    dim3 dimGrid(ceil(img_width/16.0), ceil(img_height/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    blurKernel<<<dimGrid,dimBlock>>>(d_Pout, d_Pin, img_width, img_height);

    // copy Pout from the device memory
    cudaMemcpy(out_png, d_Pout, size*CHANNELS, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i ++) {
        fprintf(stdout, "%d\t", out_png[i]);
    } printf("\n");

    // check if all zero
    non_zero_found = false;
    for (int i = 0; !non_zero_found && i < size; i ++) {
        if (out_png[i] > 0)
            non_zero_found = true;
    }
    if (non_zero_found) {
        fprintf(stderr, "[DEBUG] Non-zero pixels found in output image.\n");
    } else {
        fprintf(stderr, "[ERROR] No non-zero pixels found in output image\n");
    }

    // Free device vectors
    cudaFree(d_Pin);
    cudaFree(d_Pout);

    save_png(output_filename, out_png, img_width, img_height);
    
    // Free host memory
    free(img.data);
    free(out_png);
}
