#pragma once
#ifndef CONV2D_FUNCTIONS_H
#define CONV2D_FUNCTIONS_H

#include "conv2d_kernels.cuh"

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

void conv2d(float* M, float* F, float* P, int r, int height, int width);


#endif