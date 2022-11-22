#include <stdio.h>
#include <time.h>
#include <math.h>

// kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y){
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

// main function
int main(int argc, char *argv[]){
  int N = 1 << 27; // ~134M elements

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

  add<<<1,1>>>(N, x, y);
  cudaDeviceSynchronize();

  clock_t end = clock();
  double elapsed_seconds = double(end - start)/CLOCKS_PER_SEC/1000;

  // check for the errors
  float max_error = 3.0f;
  for (int i = 0; i < N; i++){
    max_error = fmax(max_error, fabs(y[i] - 3.0f));
  }
  printf("N=%d | Elapsed time on GPU: %10.2f ms | max_error=%7.2f\n", N, elapsed_seconds, max_error);

  // free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}