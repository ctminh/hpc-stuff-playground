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

        // generate random numbers
        curandGenerator_t curandGenerator;
        curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
        curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL) ;
        curandGenerateNormal(curandGenerator, d_normals.getData(), N_NORMALS, 0.0f, sqrdt);
        double t2=double(clock())/CLOCKS_PER_SEC;

        // call the kernel
        mc_dao_call(d_s.getData(), T, K, B, S0, sigma, mu, r, dt, d_normals.getData(), N_STEPS, N_PATHS);
        cudaDeviceSynchronize();

        // copy results from device to host
        d_s.get(&s[0], N_PATHS);

        // compute the payoff average
        double temp_sum=0.0;
        for(size_t i=0; i<N_PATHS; i++) {
            temp_sum +=s[i];
        }
        temp_sum/=N_PATHS;
        double t4=double(clock())/CLOCKS_PER_SEC;

        // init variables for CPU Monte Carlo
        vector<float> normals(N_NORMALS);
        d_normals.get(&normals[0],N_NORMALS);
        double sum=0.0;
        float s_curr=0.0;

        // CPU Monte Carlo Simulation
        for (size_t i=0; i<N_PATHS; i++) {
            int n_idx = i*N_STEPS;

            s_curr=S0;
            int n=0;

            do {
                s_curr = s_curr + mu*s_curr*dt + sigma*s_curr*normals[n_idx];
                n_idx ++;
                n++;
            }
            while (n<N_STEPS && s_curr>B);
            
            double payoff = (s_curr>K ? s_curr-K : 0.0);
            sum += exp(-r*T) * payoff;
        }

        sum/=N_PATHS;
        double t5=double(clock())/CLOCKS_PER_SEC;

        cout<<"****************** INFO ******************\n";
        cout<<"Number of Paths: " << N_PATHS << "\n";
        cout<<"Underlying Initial Price: " << S0 << "\n";
        cout<<"Strike: " << K << "\n";
        cout<<"Barrier: " << B << "\n";
        cout<<"Time to Maturity: " << T << " years\n";
        cout<<"Risk-free Interest Rate: " << r << "%\n";
        cout<<"Annual drift: " << mu << "%\n";
        cout<<"Volatility: " << sigma << "%\n";
        cout<<"****************** PRICE ******************\n";
        cout<<"Option Price (GPU): " << temp_sum << "\n";
        cout<<"Option Price (CPU): " << sum << "\n";
        cout<<"******************* TIME *****************\n";
        cout<<"GPU Monte Carlo Computation: " << (t3-t2)*1e3 << " ms\n";
        cout<<"CPU Monte Carlo Computation: " << (t5-t4)*1e3 << " ms\n";
        cout<<"******************* END *****************\n";

        // destroy generator
        curandDestroyGenerator( curandGenerator ) ;
    }
    catch(exception& e) {
        cout<< "exception: " << e.what() << "\n";
    }

    return 0;
}