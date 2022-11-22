#include <iostream>
#include <math.h>
#include <chrono>
#include <ctime>

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

// main function
int main(int argc, char *argv[]){
  int N = 1 << 27; // ~134M elements

  float *x = new float[N];
  float *y = new float[N];

  // init the arrays
  for (int i = 0; i < N; i++){
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // call and run the kernal on CPU side
  auto start = std::chrono::system_clock::now();
  add(N, x, y);
  auto end = std::chrono::system_clock::now();
  double elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  // check for the errors
  float max_error = 3.0f;
  for (int i = 0; i < N; i++){
    max_error = fmax(max_error, fabs(y[i] - 3.0f));
  }
  std::cout << "N=" << N << " | Elapsed time on CPU: " << elapsed_seconds << " ms | max_error=" << max_error << std::endl;

  // free memory
  delete [] x;
  delete [] y;

  return 0;
}