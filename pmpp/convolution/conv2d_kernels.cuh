#pragma once
#ifndef CONV2D_KERNEL_H
#define CONV2D_KERNEL_H

#define FILTER_RADIUS 1
#define BLOCK_SIZE 32

__global__ void conv2d_kernel(float* M, float* F, float* P, int r, int height, int width);

#endif