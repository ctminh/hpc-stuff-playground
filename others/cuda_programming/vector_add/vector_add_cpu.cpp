#include <iostream>
#include <math.h>

// function to add elements of 2 arrays
void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

// main function
int main(void)
{
    int N = 1 << 20; // 1 milion elements
    float *x = new float[N];
    float *y = new float[N];

    // initialize x and y
    std::cout << "1. Init array x, y" << std::endl;
    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // run add kernel on CPU
    std::cout << "2. Run add kernel on CPU" << std::endl;
    add(N, x, y);

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    delete [] x;
    delete [] y;

    return 0;
}