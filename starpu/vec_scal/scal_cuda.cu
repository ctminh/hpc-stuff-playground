#include <starpu.h>
#include <stdio.h>

static __global__ void vector_mult_cuda(unsigned n, float *val, float factor)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        val[i] *= factor;
}

extern "C" void scal_cuda_func(void *buffers[], void *_args)
{
    float *factor = (float *)_args;

    /* length of the vector */
    unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
    printf("[scal_cuda_func] scale the vector with factor = %f\n", *factor);

    /* load copy of the vector pointer */
    float *val = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
    unsigned threads_per_block = 64;
    unsigned nblocks = (n + threads_per_block - 1) / threads_per_block;
    vector_mult_cuda<<<nblocks, threads_per_block, 0, starpu_cuda_get_local_stream()>>> (n, val, *factor);
    printf("[scal_cuda_func] check the result: %0.2f\n", val[2]);
    
    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}