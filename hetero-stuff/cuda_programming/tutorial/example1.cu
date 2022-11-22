#include <iostream>
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
  auto start = std::chrono::system_clock::now();

  add<<<1,1>>>(N, x, y);
  cudaDeviceSynchronize();

  auto end = std::chrono::system_clock::now();
  double elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  // check for the errors
  float max_error = 3.0f;
  for (int i = 0; i < N; i++){
    max_error = fmax(max_error, fabs(y[i] - 3.0f));
  }
  std::cout << "N=" << N << " | Elapsed time on GPU: " << elapsed_seconds << " ms | max_error=" << max_error << std::endl;

  // free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}