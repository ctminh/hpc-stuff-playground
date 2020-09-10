/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

#include <stdlib.h>
#include <stdio.h>

#define CHECK_RESULT 1
#define ENABLE_NAIVE 1

// Thread block size
#define BLOCK_SIZE 2

// outer product vetor size is VECTOR_SIZE * BLOCK_SIZE
#define VECTOR_SIZE 2

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA (2 * BLOCK_SIZE) // Matrix A width
#define HA (2 * BLOCK_SIZE) // Matrix A height
#define WB (2 * BLOCK_SIZE) // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

void printMat(float *M, int mat_size)
{
    int i, j;
    for (i = 0; i < mat_size; i++){
        for (j = 0; j < mat_size; j++){
            printf("%.2f ", M[i*mat_size + j]);
        }
        printf("\n");
    }
}

#endif // _MATRIXMUL_H_
