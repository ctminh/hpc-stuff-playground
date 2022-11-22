#include <stdio.h>
#include <time.h>
#include <math.h>

// kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y){
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride){
    printf("arrayIdx=%d | blockIdx.x=%d, threadIdx.x=%d\n", i, blockIdx.x, threadIdx.x);
    y[i] = x[i] + y[i];
  }
}

// main function
int main(int argc, char *argv[]){
  int N = 1 << 10; // ~134M elements
  int blocks = 1;
  int threads = 1;

  // adjust the blocks and threads
  if (argc > 1){
    blocks = atoi(argv[1]);    // adjust block size
    threads = atoi(argv[2]);   // adjust num. threads/block
  }

  // allocate unified memory which is accessible from CPU or GPU
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // init the arrays, x, y on the host
  for (int i = 0; i < N; i++){
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // set timer and run the kernal on GPU side
  clock_t start = clock();

  add<<<blocks,threads>>>(N, x, y);
  cudaDeviceSynchronize();

  clock_t end = clock();
  double elapsed_seconds = double(end - start)/CLOCKS_PER_SEC;

  // check for the errors
  float max_error = 3.0f;
  for (int i = 0; i < N; i++){
    max_error = fmax(max_error, fabs(y[i] - 3.0f));
  }
  printf("---------------------------------------------------\n");
  printf("num_blocks x num_threads = %d x %d\n", blocks, threads);
  printf("N=%d | Elapsed time: %10.2f ms | max_error=%7.2f\n", N, elapsed_seconds*1000, max_error);
  printf("---------------------------------------------------\n");

  // free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}