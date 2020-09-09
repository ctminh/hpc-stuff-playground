// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "matmul.h"

// includes, kernels
// #include <matrixMul_kernel.cuh>
// #include <matrixMul_naive.cuh>
// #include <matrixMul_tiling.cuh>
// #include <matrixMul_coalescing.cuh>
// #include <matrixMul_noBankConflict.cuh>
// #include <matrixMul_compOpt.cuh>
// #include <matrixMul_unroll.cuh>
// #include <matrixMul_prefetch.cuh>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);
void randomInit(float *, int);
void printDiff(float *, float *, int, int);

////////////////////////////////////////////////////////////////////////////////
// naive version of mat mul on CPU
extern "C"
void matmul_naive_cpu(float *, const float*, const float*, unsigned int, unsigned int, unsigned int);

////////////////////////////////////////////////////////////////////////////////
// Helper Functions
////////////////////////////////////////////////////////////////////////////////
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif

inline int stringRemoveDelimiter(char delimiter, const char *string)
{
    int string_start = 0;

    while (string[string_start] == delimiter)
    {
        string_start++;
    }

    if (string_start >= (int)strlen(string)-1)
    {
        return 0;
    }

    return string_start;
}

inline bool checkCmdLineFlag(const int argc, const char **argv, const char *string_ref)
{
    bool bFound = false;

    if (argc >= 1)
    {
        for (int i=1; i < argc; i++)
        {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];

            const char *equal_pos = strchr(string_argv, '=');
            int argv_length = (int)(equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

            int length = (int)strlen(string_ref);

            if (length == argv_length && !STRNCASECMP(string_argv, string_ref, length))
            {
                bFound = true;
                continue;
            }
        }
    }

    return bFound;
}

inline int getCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref)
{
    bool bFound = false;
    int value = -1;

    if (argc >= 1)
    {
        for (int i=1; i < argc; i++)
        {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];
            int length = (int)strlen(string_ref);

            if (!STRNCASECMP(string_argv, string_ref, length))
            {
                if (length+1 <= (int)strlen(string_argv))
                {
                    int auto_inc = (string_argv[length] == '=') ? 1 : 0;
                    value = atoi(&string_argv[length + auto_inc]);
                }
                else
                {
                    value = 0;
                }

                bFound = true;
                continue;
            }
        }
    }

    if (bFound)
    {
        return value;
    }
    else
    {
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    runTest(argc, argv);

    exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv)
{
    /****************************************************/
    /*  Preparations                                    */
    /****************************************************/

    printf("[Matrix Multiply Using CUDA] - Starting...\n");
    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("    \t-wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("    \t-wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("    \tNote: Outer matrix dimensions of A & B matrices must be equal.\n");
        exit(EXIT_SUCCESS);
    }

    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    printf("1. Check device ID...\n");
    int devID = 0;
    if (checkCmdLineFlag(argc, (const char **)argv, "device")){
        devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        cudaSetDevice(devID);
    }
    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);
    if (error != cudaSuccess){
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }
    
    error = cudaGetDeviceProperties(&deviceProp, devID);
    if (deviceProp.computeMode == cudaComputeModeProhibited){
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess){
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else{
        printf("\tGPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // utilities
    cudaEvent_t start;  // get start time
    cudaEvent_t stop;   // get end time
    float msecTotal;    // runtime
    srand(2006);        // set seed for rand()

    // allocate host memory for matrices A and B
    printf("2. Allocate memory for matrix A, B, C on host machine...\n");
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);
    float flop = 2 * (float)WC * (float)HC * (float)WA;
    // allocate host memory for the result
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C = (float*) malloc(mem_size_C);

    // initialize host memory
    printf("\tInitialize matrix A, B...\n");
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    /////////////////////////////////////////////////////////////////////
    printf("3. Call the computing kernel...\n");
    /////////////////////////////////////////////////////////////////////
#if CHECK_RESULT == 1
    printf("   3.1. Matmul_Naive_CPU...\n");
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // compute reference solution
    float* reference = (float*) malloc(mem_size_C);
    matmul_naive_cpu(reference, h_A, h_B, HA, WA, WB);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("   Naive CPU (CPU-version Reference): matrix_size = %dx%d\n", HA, WA);
    printf("   Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);
#endif


}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height)
{
    int i,j,k;
    int error_count = 0;
    for (j = 0; j < height; j++) {
        for (i = 0; i < width; i++) {
            k = j*width + i;
            if (fabs(data1[k] - data2[k]) > 0.1 ) {
                printf("diff(%d,%d) CPU=%4.4f, GPU=%4.4f \n", i, j, data1[k], data2[k]);
                error_count++;
            }
        }
    }
    printf("Total Errors = %d \n", error_count);
}
