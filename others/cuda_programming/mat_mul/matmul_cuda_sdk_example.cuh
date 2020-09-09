/* Matrix multiplication: C = A * B.
 * Device code.
 */

 #ifndef _MATRIXMUL_CUDA_SDK_EXAMPLE_H_
 #define _MATRIXMUL_CUDA_SDK_EXAMPLE_H_

 #include <stdio.h>
 #include "matmul.h"

//  for what ???
#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
    #define AS(i, j) cutilBankChecker(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
    #define BS(i, j) cutilBankChecker(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
    #define AS(i, j) As[i][j]
    #define BS(i, j) Bs[i][j]
#endif

__global__ void
matmul_cuda_sdk(float *C, float *A, float *B, int wA, int wB)
{
    // block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // check matrix A & B which are stored as 1D array
    printf("\tMatrix A: \n");
    printMat(A, wA);
    printf("\tMatrix A: \n");
    printMat(B, wB);

    // index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE *by;
    int aEnd = aBegin + wA - 1;
    printf("\tCheck index of the 1st and last sub-mat of A\n");
    printf("\taBegin = wA * BLOCK_SIZE * by = %d * %d * %d = %d\n", wA, BLOCK_SIZE, by, aBegin);
    printf("\taEnd = aBegin * wA - 1 = %d * %d - 1 = %d\n", aBegin, wA, aEnd);
}

#endif // #ifndef _MATRIXMUL_CUDA_SDK_EXAMPLE_H_