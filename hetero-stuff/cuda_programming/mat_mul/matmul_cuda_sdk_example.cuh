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

    // Put "the declaration of the shared memory array" out of the loop
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE *by;
    int aEnd = aBegin + wA - 1;
    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;
    // printf("\tCheck index of the 1st and last sub-mat of A\n");
    // printf("\taBegin = wA * BLOCK_SIZE * by = %d * %d * %d = %d\n", wA, BLOCK_SIZE, by, aBegin);
    // printf("\taEnd = aBegin * wA - 1 = %d * %d - 1 = %d\n", aBegin, wA, aEnd);

    // index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;
    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep)
    {
        // Declaration of the shared memory array As used to store the sub-matrix of A
        // __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to store the sub-matrix of B
        // __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        AS(ty, tx) = A[a + wA*ty + tx];
        BS(ty, tx) = B[b + wB*ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together each thread computes one element of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory; each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB*ty + tx] = Csub;

}

#endif // #ifndef _MATRIXMUL_CUDA_SDK_EXAMPLE_H_