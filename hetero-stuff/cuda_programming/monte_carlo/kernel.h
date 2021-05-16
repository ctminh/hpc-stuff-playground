#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_

void mc_dao_call(float * d_s, float T, float K, float B, float S0, float sigma, float mu, float r, float dt, float* d_normals, unsigned N_STEPS, unsigned N_PATHS);

#endif