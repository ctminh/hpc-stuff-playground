#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "dev_array.h"
#include <curand.h>

/* Underlying Price Simulation */
/* For simulating the underlying price, we must discretise the underlying's changes.
    + using the Euler method, a good approximation
    + dS(t) = mu.S(t).dt + sigma.S(t).dW(t)
if we see this as a finite difference, we can rewrite everything in the following way:
    + Y(n+1) - Y(n) = mu.Y(n).delta(T) + sigma.Y(n).delta(Wt)
where,
    mu:     the expected return per year
    sigma:  the expected volatility per year
    T:      is the time to maturity
    dt:     is the amount of time elapsing at each step
    S(t):   is the current price at t
    dW:     is a random number distributed according to a normal distribution with mean 0 and 
    Y:      is the price at the time step n, where Yo = So
*/

using namespace std;

int main(int argc, char *argv[])
{
    try {
        // declare variables and constants
        const size_t N_PATHS = 5000000; // path:
        const size_t N_STEPS = 365;     // step:
        const size_t N_NORMALS = N_PATHS*N_STEPS;   // normal:

        const float T = 1.0f;   
        const float K = 100.0f;
        const float B = 95.0f;
        const float S0 = 100.0f;
        const float sigma = 0.2f;
        const float mu = 0.1f;
        const float r = 0.05f;

        float dt = float(T) / float(N_STEPS);
        float sqrdt = sqrt(dt);

        // generate arrays
        vector<float> s(N_PATHS);
        dev_array<float> d_s(N_PATHS);
        dev_array<float> d_normals(N_NORMALS);
    }

    return 0;
}