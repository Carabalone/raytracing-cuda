#ifndef UTILITY_H
#define UTILITY_H
#include <curand_kernel.h>
#include <cstdlib>
#include <iostream>

// returns from [0, 1[
__device__ float inline gpu_rand(curandState &state) {
    return curand_uniform(&state);
}

__device__ float inline gpu_rand(curandState &state, float min, float max) {
    return min + (max - min) * curand_uniform(&state);
}

inline double random_double() {
    return rand() / (RAND_MAX + 1.0);
}

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )


#endif // UTILITY_H
