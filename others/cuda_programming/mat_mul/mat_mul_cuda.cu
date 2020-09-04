/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

 // System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>


/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB){
    // block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;
    printf("aBegin = wA * BLOCK_SIZE * by = %d * %d * %d = %d\n", wA, BLOCK_SIZE, by, aBegin);
    
    // index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // loop over all the sub-matrices of A & B, required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b+= bStep){
        // declaration of the shared memory array As used to store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // declaration of the shared memory array Bs used to store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // load the matrices from device mem to shared mem, each thread loads 1 element of each matrix
        As[ty][tx] = A[a + wA*ty + tx];
        Bs[ty][tx] = B[b + wB*ty + tx];

        // synchronize to make sure the matrices are loaded
        __syncthreads();
    }
}

/**
 * initialize values for matrix
 */
void ConstantInit(float *data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}


/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char **argv, int block_size, const dim3 &dimsA, const dim3 &dimsB) {
    // allocate host memory for matrix A, B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = reinterpret_cast<float *>(malloc(mem_size_A));
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = reinterpret_cast<float *>(malloc(mem_size_B));
    cudaStream_t stream;

    // initialize host memory
    const float valA = 1.0f;
    const float valB = 0.01f;
    ConstantInit(h_A, size_A, valA);
    ConstantInit(h_B, size_B, valB);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = reinterpret_cast<float *>(malloc(mem_size_C));
    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }
    // check mem allocation
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

    // allocate cuda events that we will use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // copy host mem to device
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x/threads.x, dimsA.y/threads.y);

    // create and start timer
    printf("Computing result using CUDA kernel...\n");
    printf("\t Threads: %d x %d\n", block_size, block_size);
    printf("\t Grid: %d x %d\n", dimsB.x/threads.x, dimsA.y/threads.y);

    // warm up operations
    if (block_size == 16){
        MatrixMulCUDA<16> <<< grid, threads, 0, stream >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }else{
        MatrixMulCUDA<4> <<< grid, threads, 0, stream >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
    printf("Done!\n");

    // check cuda stream error
    checkCudaErrors(cudaStreamSynchronize(stream));

    // record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    // execute the kernel
    int nIter = 300;
    for (int j = 0; j < nIter; j++){
        if (block_size == 16) {
            MatrixMulCUDA<16> <<< grid, threads, 0, stream >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        } else {
            MatrixMulCUDA<4> <<< grid, threads, 0, stream >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) * static_cast<double>(dimsA.y) * static_cast<double>(dimsB.x);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul,
            threads.x * threads.y);

    return 0;
}


/**
 * Program main
 */
int main(int argc, char **argv) {
    printf("[Matrix Multiply Using CUDA] - Starting...\n");
    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("\t-wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("\t-wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("\t-bs=BlockSize)\n");
        printf("Note: Outer matrix dimensions of A & B matrices must be equal.\n");
        exit(EXIT_SUCCESS);
    }

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char **)argv);

    // declare the matrix
    int block_size = 4;
    dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);
    // get width of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "wA")){
        dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
    }
    // get height of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "hA")){
        dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
    }
    // get height of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "wB")){
        dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
    }
    // get height of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "hB")){
        dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
    }
    // get block size
    if (checkCmdLineFlag(argc, (const char **)argv, "bs")){
        block_size = getCmdLineArgumentInt(argc, (const char **)argv, "bs");
    }
    // check the size of A & B
    if (dimsA.x != dimsB.y){
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n", dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
    
    int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);

    exit(matrix_result);

    return 0;
}